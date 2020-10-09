#include <index.h>
#include <future>
#include <Neighbor_Tag.h>
#include <numeric>
#include <omp.h>
#include <shard.h>
#include <string.h>
#include <sync_index.h>
#include <time.h>
#include <timer.h>
#include <cstring>
#include <iomanip>

#include "aux_utils.h"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "memory_mapper.h"

#define NUM_INSERT_THREADS 16
#define NUM_DELETE_THREADS 4
#define NUM_SEARCH_THREADS 4

template<typename T, typename TagT>
void sync_search_kernel(T* query, size_t query_num, size_t query_aligned_dim,
                        const int recall_at, _u64    L,
                        diskann::SyncIndex<T, TagT>& sync_index,
                        const std::string&    truthset_file,
                        tsl::robin_set<TagT>& inactive_tags) {
  unsigned* gt_ids = NULL;
  float*    gt_dists = NULL;
  size_t    gt_num, gt_dim;
  diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);

  //  query_num = 1;

  float* query_result_dists = new float[recall_at * query_num];
  TagT*  query_result_tags = new TagT[recall_at * query_num];
  //  memset(query_result_dists, 0, sizeof(float) * recall_at * query_num);
  //  memset(query_result_tags, 0, sizeof(TagT) * recall_at * query_num);

  for (_u32 q = 0; q < query_num; q++) {
    for (_u32 r = 0; r < (_u32) recall_at; r++) {
      query_result_tags[q * recall_at + r] = std::numeric_limits<TagT>::max();
      query_result_dists[q * recall_at + r] = std::numeric_limits<float>::max();
    }
  }

  std::vector<double> latency_stats(query_num, 0);
  std::string         recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(18)
            << "Mean Latency (ms)" << std::setw(15) << "99.9 Latency"
            << std::setw(12) << recall_string << std::endl;
  std::cout << "==============================================================="
               "==============="
            << std::endl;
  auto      s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(NUM_SEARCH_THREADS)
  for (int64_t i = 0; i < (int64_t) query_num; i++) {
    auto qs = std::chrono::high_resolution_clock::now();
    sync_index.search_async(query + i * query_aligned_dim, recall_at, L,
                            query_result_tags + i * recall_at,
                            query_result_dists + i * recall_at);

    auto qe = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = qe - qs;
    latency_stats[i] = diff.count() * 1000;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    /*      for (_u32 t = 0;t < (_u32) recall_at; t++) {
            std::cout<<t<<": " << query_result_tags[i*recall_at + t] << ","
       <<query_result_dists[i*recall_at +t] <<" ";
        }
        std::cout<<std::endl;
        */
  }
  auto e = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = e - s;
  float                         qps = (query_num / diff.count());

  float recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                           query_result_tags, recall_at,
                                           recall_at, inactive_tags);

  std::sort(latency_stats.begin(), latency_stats.end());
  std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
            << std::accumulate(latency_stats.begin(), latency_stats.end(), 0) /
                   (float) query_num
            << std::setw(15) << (float) latency_stats[(_u64)(0.999 * query_num)]
            << std::setw(12) << recall << std::endl;

  delete[] query_result_dists;
  delete[] query_result_tags;
}

template<typename T, typename TagT>
void insertion_kernel(T* data_load, diskann::SyncIndex<T, TagT>& sync_index,
                      std::vector<TagT>& insert_vec, size_t aligned_dim) {
  diskann::Timer timer;
#pragma omp      parallel for num_threads(NUM_INSERT_THREADS)
  for (_s64 i = 0; i < (_s64) insert_vec.size(); i++) {
    sync_index.insert(data_load + aligned_dim * insert_vec[i], insert_vec[i]);
  }
  float time_secs = timer.elapsed() / 1.0e6f;
  std::cout << "Inserted " << insert_vec.size() << " points in " << time_secs
            << "s" << std::endl;
}

template<typename T, typename TagT>
void delete_kernel(diskann::SyncIndex<T, TagT>& sync_index,
                   std::vector<TagT>& delete_vec) {
  diskann::Timer timer;
#pragma omp      parallel for num_threads(NUM_DELETE_THREADS)
  for (size_t i = 0; i < delete_vec.size(); ++i)
    sync_index.lazy_delete(delete_vec[i]);
  float time_secs = timer.elapsed() / 1.0e6f;
  std::cout << "Deleted " << delete_vec.size() << " points in " << time_secs
            << "s" << std::endl;
}

template<typename T, typename TagT>
void test(const std::string& data_path, const unsigned L_mem,
          const unsigned R_mem, const float alpha_mem, const unsigned L_disk,
          const unsigned R_disk, const float alpha_disk, const size_t num_incr,
          const size_t num_del, const size_t num_shards,
          const unsigned num_pq_chunks, const unsigned nodes_to_cache,
          const std::string& save_path, const std::string& query_file,
          const std::string& truthset_file, const int recall_at, _u64 Lsearch,
          const unsigned beam_width) {
  diskann::Parameters paras;
  paras.Set<unsigned>("L_mem", L_mem);
  paras.Set<unsigned>("R_mem", R_mem);
  paras.Set<float>("alpha_mem", alpha_mem);
  paras.Set<unsigned>("L_disk", L_disk);
  paras.Set<unsigned>("R_disk", R_disk);
  paras.Set<float>("alpha_disk", alpha_disk);
  paras.Set<unsigned>("C", 1500);
  paras.Set<unsigned>("beamwidth", beam_width);
  paras.Set<unsigned>("num_pq_chunks", num_pq_chunks);
  paras.Set<unsigned>("nodes_to_cache", nodes_to_cache);
  T*     data_load = NULL;
  size_t num_points, dim, aligned_dim;

  diskann::load_aligned_bin<T>(data_path.c_str(), data_load, num_points, dim,
                               aligned_dim);

  std::cout << "Loaded full data for driver." << std::endl;
  diskann::SyncIndex<T, TagT> sync_index(num_points + 5000, dim, num_shards,
                                         paras, 2, save_path);
  size_t res = sync_index.load(save_path);
  std::cout << "Sync index loaded " << res << " points" << std::endl;

  T*     query = NULL;
  size_t query_num, query_dim, query_aligned_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);

  tsl::robin_set<TagT> inactive_points;
  for (_u64 p = res; p < num_points; p++)
    inactive_points.insert(p);

  std::cout << "Searching before inserts and deletes: " << std::endl;

  sync_search_kernel(query, query_num, query_aligned_dim, recall_at, Lsearch,
                     sync_index, truthset_file, inactive_points);

  std::random_device                      rd;
  std::mt19937                            gen(rd());
  std::uniform_int_distribution<unsigned> dis(0, res - 1);
  tsl::robin_set<TagT>                    delete_list;
  std::vector<TagT>                       delete_vec;
  std::vector<unsigned>                   insert_vec;
  while (delete_list.size() < num_del)
    delete_list.insert(dis(gen));
  for (auto p : delete_list)
    delete_vec.push_back(p);
  for (unsigned i = res; i < (res + num_incr); i++)
    insert_vec.push_back(i);

  std::cout << "delete_list, vec size: " << delete_list.size() << " "
            << delete_vec.size() << std::endl;

  {
    std::future<void> insert_future =
        std::async(std::launch::async, insertion_kernel<T, TagT>, data_load,
                   std::ref(sync_index), std::ref(insert_vec), aligned_dim);

    std::future<void> delete_future =
        std::async(std::launch::async, delete_kernel<T, TagT>,
                   std::ref(sync_index), std::ref(delete_vec));

    std::future_status insert_status, delete_status;

    tsl::robin_set<TagT> dummy_set;
    //    query_num  = 1;
    unsigned total_queries = 0;
    do {
      insert_status = insert_future.wait_for(std::chrono::milliseconds(1));
      delete_status = delete_future.wait_for(std::chrono::milliseconds(1));

      if (insert_status == std::future_status::deferred ||
          delete_status == std::future_status::deferred) {
        std::cout << "deferred\n";
      } else if (insert_status == std::future_status::timeout ||
                 delete_status == std::future_status::timeout) {
        sync_search_kernel(query, query_num, query_aligned_dim, recall_at,
                           Lsearch, sync_index, truthset_file, dummy_set);
        total_queries += query_num;
        std::cout << "Queries processed: " << total_queries << std::endl;
      }
      if (insert_status == std::future_status::ready) {
        std::cout << "Insertions complete!\n";
      }
      if (delete_status == std::future_status::ready) {
        std::cout << "Deletions complete!\n";
      }
    } while (insert_status != std::future_status::ready ||
             delete_status != std::future_status::ready);
  }

  std::cout << "delete_list, delete vec size: " << delete_list.size() << " "
            << delete_vec.size() << std::endl;

  tsl::robin_set<TagT> dummy_set;
  for (auto elem : delete_vec)
    dummy_set.insert(elem);
  for (unsigned i = (res + num_incr); i < num_points; i++)
    dummy_set.insert(i);

  std::cout << "Search after inserts and deletes, but before merge:"
            << std::endl;
  sync_search_kernel(query, query_num, query_aligned_dim, recall_at, Lsearch,
                     sync_index, truthset_file, dummy_set);

  sync_index.merge_all(save_path);
  std::cout << "Merge complete. Now searching..." << std::endl;
  sync_search_kernel(query, query_num, query_aligned_dim, recall_at, Lsearch,
                     sync_index, truthset_file, dummy_set);
  delete[] data_load;
}

int main(int argc, char** argv) {
  if (argc < 20) {
    std::cout
        << "Correct usage: " << argv[0]
        << " <type[int8/uint8/float]> <data_file> <L_mem> <R_mem> <alpha_mem>"
        << " <L_disk> <R_disk> <alpha_disk>"
        << " <num_incr> <num_del> <num_shards> <#pq_chunks> <#nodes_to_cache>"
        << " <save_graph_file> <query_file> <truthset_file> <recall@> "
           "<Lsearch> <#beam_width>"
        << std::endl;
    exit(-1);
  }

  int         arg_no = 3;
  unsigned    L_mem = (unsigned) atoi(argv[arg_no++]);
  unsigned    R_mem = (unsigned) atoi(argv[arg_no++]);
  float       alpha_mem = (float) std::atof(argv[arg_no++]);
  unsigned    L_disk = (unsigned) atoi(argv[arg_no++]);
  unsigned    R_disk = (unsigned) atoi(argv[arg_no++]);
  float       alpha_disk = (float) std::atof(argv[arg_no++]);
  size_t      num_incr = (size_t) std::atoi(argv[arg_no++]);
  size_t      num_del = (size_t) std::atoi(argv[arg_no++]);
  size_t      num_shards = (size_t) std::atoi(argv[arg_no++]);
  unsigned    num_pq_chunks = (unsigned) std::atoi(argv[arg_no++]);
  unsigned    nodes_to_cache = (unsigned) std::atoi(argv[arg_no++]);
  std::string save_path(argv[arg_no++]);
  std::string query_file(argv[arg_no++]);
  std::string truthset(argv[arg_no++]);
  int         recall_at = (int) std::atoi(argv[arg_no++]);
  _u64        Lsearch = std::atoi(argv[arg_no++]);
  unsigned    beam_width = (unsigned) std::atoi(argv[arg_no++]);

  if (std::string(argv[1]) == std::string("int8"))
    test<int8_t, unsigned>(argv[2], L_mem, R_mem, alpha_mem, L_disk, R_disk,
                           alpha_disk, num_incr, num_del, num_shards,
                           num_pq_chunks, nodes_to_cache, save_path, query_file,
                           truthset, recall_at, Lsearch, beam_width);
  else if (std::string(argv[1]) == std::string("uint8"))
    test<uint8_t, unsigned>(
        argv[2], L_mem, R_mem, alpha_mem, L_disk, R_disk, alpha_disk, num_incr,
        num_del, num_shards, num_pq_chunks, nodes_to_cache, save_path,
        query_file, truthset, recall_at, Lsearch, beam_width);
  else if (std::string(argv[1]) == std::string("float"))
    test<float, unsigned>(argv[2], L_mem, R_mem, alpha_mem, L_disk, R_disk,
                          alpha_disk, num_incr, num_del, num_shards,
                          num_pq_chunks, nodes_to_cache, save_path, query_file,
                          truthset, recall_at, Lsearch, beam_width);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
