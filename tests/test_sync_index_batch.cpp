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

#define NUM_INSERT_THREADS 31
#define NUM_DELETE_THREADS 1
#define NUM_SEARCH_THREADS 30

template<typename T, typename TagT>
void sync_search_kernel(
    T* query, size_t query_num, size_t query_aligned_dim, const int recall_at,
    std::vector<_u64> Lvec, diskann::SyncIndex<T, TagT>& sync_index,
    const std::string&       truthset_file,
    tsl::robin_set<unsigned> delete_list = tsl::robin_set<unsigned>()) {
  unsigned* gt_ids = NULL;
  float*    gt_dists = NULL;
  size_t    gt_num, gt_dim;
  diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);

  //  _u64 rnd_query = rand() % query_num;

  //  std::cout<<rnd_query << ":" << std::flush;
  //  query_num = 1;
  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(18)
            << "Mean Latency (ms)" << std::setw(15) << "99.9 Latency";

  if (delete_list.size() != 0) {
    std::cout << std::setw(12) << recall_string << std::endl;
  } else
    std::cout << std::endl;

  std::cout << "==============================================================="
               "==============="
            << std::endl;

  float* query_result_dists = new float[recall_at * query_num];
  TagT*  query_result_tags = new TagT[recall_at * query_num];
  memset(query_result_dists, 0, sizeof(float) * recall_at * query_num);
  memset(query_result_tags, 0, sizeof(TagT) * recall_at * query_num);

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    _u64                L = Lvec[test_id];
    std::vector<double> latency_stats(query_num, 0);
    auto                s = std::chrono::high_resolution_clock::now();
#pragma omp             parallel for num_threads(NUM_SEARCH_THREADS)
    for (int64_t i = 0; i < (int64_t) query_num; i++) {
      auto qs = std::chrono::high_resolution_clock::now();
      sync_index.search_async(query + i * query_aligned_dim, recall_at,
                              (_u32) L, query_result_tags + i * recall_at,
                              query_result_dists + i * recall_at);

      auto qe = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = qe - qs;
      latency_stats[i] = diff.count() * 1000;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    auto e = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = e - s;
    float                         qps = (float) (query_num / diff.count());

    float recall = (float) diskann::calculate_recall(
        (_u32) query_num, gt_ids, gt_dists, (_u32) gt_dim, query_result_tags,
        (_u32) recall_at, (_u32) recall_at, delete_list);

    std::sort(latency_stats.begin(), latency_stats.end());

    std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
              << std::accumulate(latency_stats.begin(), latency_stats.end(),
                                 0) /
                     (float) query_num
              << std::setw(15)
              << (float) latency_stats[(_u64)(0.999 * query_num)];
    if (delete_list.size() != 0) {
      std::cout << std::setw(12) << recall << std::endl;
    } else
      std::cout << std::endl;
  }

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
  for (_s64 i = 0; i < (_s64) delete_vec.size(); ++i) {
    sync_index.lazy_delete(delete_vec[i]);
    //    std::this_thread::sleep_for(std::chrono::microseconds(12));
  }
  float time_secs = timer.elapsed() / 1.0e6f;
  std::cout << "Deleted " << delete_vec.size() << " points in " << time_secs
            << "s" << std::endl;
}

template<typename T, typename TagT>
void test(const std::string& data_path, const unsigned L_mem,
          const unsigned R_mem, const float alpha_mem, const unsigned L_disk,
          const unsigned R_disk, const float alpha_disk,
          const size_t num_incr_batch, const size_t num_del_batch,
          const size_t num_batches, const size_t num_shards,
          const unsigned num_pq_chunks, const unsigned nodes_to_cache,
          const std::string& save_path, const std::string& query_file,
          const std::string& truthset_file, const int recall_at,
          std::vector<_u64> Lvec, const unsigned beam_width) {
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

  if (1.1 * num_incr_batch > SHORT_MAX_POINTS * num_shards) {
    std::cout << "please provide " << num_incr_batch << " at most "
              << (double) (SHORT_MAX_POINTS * num_shards) / 1.1 << " if using "
              << num_shards << " shards." << std::endl;
    exit(-1);
  }

  diskann::SyncIndex<T, TagT> sync_index(num_points + 5000, dim, num_shards,
                                         paras, 2, save_path);

  size_t res = sync_index.load(save_path);
  std::cout << "Sync index loaded " << res << " points" << std::endl;

  tsl::robin_set<TagT> active_tags;
  sync_index.get_active_tags(active_tags);
  if (active_tags.size() != res) {
    std::cout << "#tags loaded != #points loaded...exiting" << std::endl;
    exit(-1);
  }

  tsl::robin_set<TagT> inactive_tags;
  for (_u64 p = 0; p < num_points; p++) {
    if (active_tags.find((TagT) p) == active_tags.end())
      inactive_tags.insert((TagT) p);
  }
  if ((active_tags.size() + inactive_tags.size()) != num_points) {
    std::cout << "Error in size of active tags and inactive tags.   : "
              << active_tags.size() << "  ,  " << inactive_tags.size()
              << std::endl;
    exit(-1);
  }

  T*     query = NULL;
  size_t query_num, query_dim, query_aligned_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);
  std::cout << "Search before any insertions/deletions." << std::endl;
  sync_search_kernel(query, query_num, query_aligned_dim, recall_at, Lvec,
                     sync_index, truthset_file, inactive_tags);

  tsl::robin_set<TagT> new_active_tags;
  tsl::robin_set<TagT> new_inactive_tags;

  _u64 batch_id = 0;
  while (batch_id++ < num_batches) {
    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_real_distribution<float> dis(0, 1);
    std::vector<TagT>                     delete_vec;
    std::vector<TagT>                     insert_vec;

    new_active_tags.clear();
    new_inactive_tags.clear();

    delete_vec.clear();
    insert_vec.clear();

    float active_tags_sampling_rate = (float) ((std::min)(
        (1.0 * num_del_batch) / (1.0 * active_tags.size()), 1.0));

    for (auto iter = active_tags.begin(); iter != active_tags.end(); iter++) {
      if (dis(gen) < active_tags_sampling_rate) {
        delete_vec.emplace_back(*iter);
        new_inactive_tags.insert(*iter);
      } else
        new_active_tags.insert(*iter);
    }

    float inactive_tags_sampling_rate = (float) ((std::min)(
        (1.0 * num_incr_batch) / (1.0 * inactive_tags.size()), 1.0));

    for (auto iter = inactive_tags.begin(); iter != inactive_tags.end();
         iter++) {
      if (dis(gen) < inactive_tags_sampling_rate) {
        insert_vec.emplace_back(*iter);
        new_active_tags.insert(*iter);
      } else
        new_inactive_tags.insert(*iter);
    }

    std::cout << "Preparing to insert " << insert_vec.size()
              << " points and delete  " << delete_vec.size() << " points. "
              << std::endl;

    {
      std::future<void> insert_future =
          std::async(std::launch::async, insertion_kernel<T, TagT>, data_load,
                     std::ref(sync_index), std::ref(insert_vec), aligned_dim);

      std::future<void> delete_future =
          std::async(std::launch::async, delete_kernel<T, TagT>,
                     std::ref(sync_index), std::ref(delete_vec));

      std::future_status insert_status, delete_status;
      //      _u64               total_queries = 0;
      do {
        insert_status = insert_future.wait_for(std::chrono::milliseconds(1));
        delete_status = delete_future.wait_for(std::chrono::milliseconds(1));

        if (insert_status == std::future_status::deferred ||
            delete_status == std::future_status::deferred) {
          std::cout << "deferred\n";
        } else if (insert_status == std::future_status::timeout ||
                   delete_status == std::future_status::timeout) {
          // sync_search_kernel(query, query_num, query_aligned_dim,
          // recall_at,Lsearch, sync_index, truthset_file);
          //           total_queries += query_num;
          //         std::cout << "Queries processed: " << total_queries <<
          //         std::endl;
        }
        if (insert_status == std::future_status::ready) {
          // std::cout << "Insertions complete!\n";
        }
        if (delete_status == std::future_status::ready) {
          // std::cout << "Deletions complete!\n";
        }
      } while (insert_status != std::future_status::ready ||
               delete_status != std::future_status::ready);
    }

    std::cout << "Inserts and deletes in batch complete." << std::endl;

    inactive_tags.swap(new_inactive_tags);
    active_tags.swap(new_active_tags);

    sync_index.merge_all(save_path);
    std::cout << "Merge complete. Now searching.." << std::endl;
    sync_search_kernel(query, query_num, query_aligned_dim, recall_at, Lvec,
                       sync_index, truthset_file, inactive_tags);

    std::cout << "Index size: " << sync_index.return_nd()
              << ", #active: " << active_tags.size()
              << ", #inactive: " << inactive_tags.size() << std::endl;
  }
  delete[] data_load;
}

int main(int argc, char** argv) {
  if (argc < 20) {
    std::cout << "Correct usage: " << argv[0]
              << " <type[int8/uint8/float]> <data_file> <L_mem> <R_mem> "
                 "<alpha_mem> <L_disk> <R_disk> <alpha_disk>"
              << " <num_incr_per_batch> "
                 "<num_del_per_batch> <num_batches> <num_shards> <#pq_chunks> "
                 "<#nodes_to_cache>"
              << " <save_graph_file> <query_file> <truthset_file> <recall@> "
                 "<#beam_width> <L1> <L2> <L3> ...."
              << std::endl;
    exit(-1);
  }

  unsigned ctr = 3;

  unsigned    L_mem = (unsigned) atoi(argv[ctr++]);
  unsigned    R_mem = (unsigned) atoi(argv[ctr++]);
  float       alpha_mem = (float) std::atof(argv[ctr++]);
  unsigned    L_disk = (unsigned) atoi(argv[ctr++]);
  unsigned    R_disk = (unsigned) atoi(argv[ctr++]);
  float       alpha_disk = (float) std::atof(argv[ctr++]);
  size_t      num_incr = (size_t) std::atoi(argv[ctr++]);
  size_t      num_del = (size_t) std::atoi(argv[ctr++]);
  size_t      num_batches = (size_t) std::atoi(argv[ctr++]);
  size_t      num_shards = (size_t) std::atoi(argv[ctr++]);
  unsigned    num_pq_chunks = (unsigned) std::atoi(argv[ctr++]);
  unsigned    nodes_to_cache = (unsigned) std::atoi(argv[ctr++]);
  std::string save_path(argv[ctr++]);
  std::string query_file(argv[ctr++]);
  std::string truthset(argv[ctr++]);
  int         recall_at = (int) std::atoi(argv[ctr++]);
  unsigned    beam_width = (unsigned) std::atoi(argv[ctr++]);

  std::vector<_u64> Lvec;
  for (int ctr = 20; ctr < argc; ctr++) {
    _u64 curL = std::atoi(argv[ctr]);
    if (curL >= recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    std::cout << "No valid Lsearch found. Lsearch must be at least recall_at."
              << std::endl;
    return -1;
  }
  if (std::string(argv[1]) == std::string("int8"))
    test<int8_t, unsigned>(argv[2], L_mem, R_mem, alpha_mem, L_disk, R_disk,
                           alpha_disk, num_incr, num_del, num_batches,
                           num_shards, num_pq_chunks, nodes_to_cache, save_path,
                           query_file, truthset, recall_at, Lvec, beam_width);
  else if (std::string(argv[1]) == std::string("uint8"))
    test<uint8_t, unsigned>(
        argv[2], L_mem, R_mem, alpha_mem, L_disk, R_disk, alpha_disk, num_incr,
        num_del, num_batches, num_shards, num_pq_chunks, nodes_to_cache,
        save_path, query_file, truthset, recall_at, Lvec, beam_width);
  else if (std::string(argv[1]) == std::string("float"))
    test<float, unsigned>(argv[2], L_mem, R_mem, alpha_mem, L_disk, R_disk,
                          alpha_disk, num_incr, num_del, num_batches,
                          num_shards, num_pq_chunks, nodes_to_cache, save_path,
                          query_file, truthset, recall_at, Lvec, beam_width);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
