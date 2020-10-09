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
                      diskann::Parameters& paras, size_t num_points,
                      size_t num_start, size_t aligned_dim) {
  diskann::Timer timer;
#pragma omp      parallel for num_threads(NUM_INSERT_THREADS)
  for (size_t i = num_start; i < num_points; i++) {
    sync_index.insert(data_load + aligned_dim * i, i, paras);
  }
  float time_secs = timer.elapsed() / 1.0e6f;
  std::cout << "Inserted " << num_points - num_start << " points in "
            << time_secs << "s" << std::endl;
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
          const unsigned R_disk, const float alpha_disk, const size_t num_start,
          const size_t num_shards, const unsigned num_pq_chunks,
          const unsigned nodes_to_cache, const std::string& save_path) {
  diskann::Parameters paras;
  paras.Set<unsigned>("L_mem", L_mem);
  paras.Set<unsigned>("R_mem", R_mem);
  paras.Set<float>("alpha_mem", alpha_mem);
  paras.Set<unsigned>("L_disk", L_disk);
  paras.Set<unsigned>("R_disk", R_disk);
  paras.Set<float>("alpha_disk", alpha_disk);
  paras.Set<unsigned>("C", 1500);
  paras.Set<unsigned>("beamwidth", 5);
  paras.Set<unsigned>("num_pq_chunks", num_pq_chunks);
  paras.Set<unsigned>("nodes_to_cache", nodes_to_cache);

  T*     data_load = NULL;
  size_t num_points, dim, aligned_dim;

  diskann::load_aligned_bin<T>(data_path.c_str(), data_load, num_points, dim,
                               aligned_dim);

  std::cout << "Loaded full data for driver." << std::endl;
  diskann::SyncIndex<T, TagT> sync_index(num_points + 5000, dim, num_shards,
                                         paras, 2, save_path);
  std::cout << "Ran constructor." << std::endl;
  std::vector<TagT> tags(num_start);
  std::iota(tags.begin(), tags.end(), 0);
  diskann::Timer timer;
  sync_index.build(data_path.c_str(), num_start, tags);
  std::cout << "Sync Index build time: " << timer.elapsed() / 1000000 << "s\n";

  delete[] data_load;
}

int main(int argc, char** argv) {
  if (argc < 14) {
    std::cout
        << "Correct usage: " << argv[0]
        << " <type[int8/uint8/float]> <data_file> <L_mem> <R_mem> <alpha_mem>"
        << " <L_disk> <R_disk> <alpha_disk>"
        << " <num_start> <num_shards> <#pq_chunks> <#nodes_to_cache>"
        << " <save_graph_file>" << std::endl;
    exit(-1);
  }

  int         arg_no = 3;
  unsigned    L_mem = (unsigned) atoi(argv[arg_no++]);
  unsigned    R_mem = (unsigned) atoi(argv[arg_no++]);
  float       alpha_mem = (float) std::atof(argv[arg_no++]);
  unsigned    L_disk = (unsigned) atoi(argv[arg_no++]);
  unsigned    R_disk = (unsigned) atoi(argv[arg_no++]);
  float       alpha_disk = (float) std::atof(argv[arg_no++]);
  size_t      num_start = (size_t) std::atoi(argv[arg_no++]);
  size_t      num_shards = (size_t) std::atoi(argv[arg_no++]);
  unsigned    num_pq_chunks = (unsigned) std::atoi(argv[arg_no++]);
  unsigned    nodes_to_cache = (unsigned) std::atoi(argv[arg_no++]);
  std::string save_path(argv[arg_no++]);

  if (std::string(argv[1]) == std::string("int8"))
    test<int8_t, unsigned>(argv[2], L_mem, R_mem, alpha_mem, L_disk, R_disk,
                           alpha_disk, num_start, num_shards, num_pq_chunks,
                           nodes_to_cache, save_path);
  else if (std::string(argv[1]) == std::string("uint8"))
    test<uint8_t, unsigned>(argv[2], L_mem, R_mem, alpha_mem, L_disk, R_disk,
                            alpha_disk, num_start, num_shards, num_pq_chunks,
                            nodes_to_cache, save_path);
  else if (std::string(argv[1]) == std::string("float"))
    test<float, unsigned>(argv[2], L_mem, R_mem, alpha_mem, L_disk, R_disk,
                          alpha_disk, num_start, num_shards, num_pq_chunks,
                          nodes_to_cache, save_path);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
