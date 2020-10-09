#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <omp.h>
#include <random>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <string>
#include "tcmalloc/malloc_extension.h"
#include "tsl/robin_set.h"

//#include <boost/dynamic_bitset.hpp>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include "exceptions.h"
#include "index.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "parameters.h"
#include "partition_and_pq.h"
#include "sync_index.h"
#include "shard.h"
#include "timer.h"
#include "utils.h"

#include "Neighbor_Tag.h"
#ifdef _WINDOWS
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#else
#include "linux_aligned_file_reader.h"
#endif

namespace diskann {

  // Initialize a sync_index, load the data of type T with filename
  // (bin), initialize _max_points, _num_points AND _num_shards, initialize
  // shards
  template<typename T, typename TagT>
  SyncIndex<T, TagT>::SyncIndex(const size_t max_points, const size_t dim,
                                const size_t num_shards, Parameters &parameters,
                                const unsigned     num_search_threads_per_shard,
                                const std::string &save_index_path)
      : _num_shards(num_shards), _dim(dim), _max_points(max_points) {
    _aligned_dim = ROUND_UP(_dim, 8);

    _shards.resize(_num_shards);
    size_t max_points_shard =
        (_u64)((_max_points / _num_shards) * SHARD_SLACK_FACTOR);
    for (size_t i = 0; i < _num_shards; i++) {
      _shards[i] = new diskann::Shard<T, TagT>(
          parameters, max_points_shard, _aligned_dim,
          num_search_threads_per_shard, save_index_path + std::to_string(i));
    }
  }

  template<>
  SyncIndex<float, int>::~SyncIndex() {
    //    aligned_free(_sync_data);
  }

  template<>
  SyncIndex<float, unsigned>::~SyncIndex() {
    //    aligned_free(_sync_data);
  }

  template<>
  SyncIndex<_s8, int>::~SyncIndex() {
    //    aligned_free(_sync_data);
  }

  template<>
  SyncIndex<_s8, unsigned>::~SyncIndex() {
    //    aligned_free(_sync_data);
  }

  template<>
  SyncIndex<_u8, int>::~SyncIndex() {
    //    aligned_free(_sync_data);
  }

  template<>
  SyncIndex<_u8, unsigned>::~SyncIndex() {
    //    aligned_free(_sync_data);
  }

  template<typename T, typename TagT>
  void SyncIndex<T, TagT>::build(const char *filename,
                                 const size_t             num_points_to_load,
                                 const std::vector<TagT> &tags) {
    if (filename == nullptr) {
      std::cout << "Starting with an empty index." << std::endl;
      _num_points = 0;
    } else if (!file_exists(filename)) {
      std::cerr << "Data file provided does not exist!!! Exiting...."
                << std::endl;
      exit(-1);
    }
    T *    sync_data = nullptr;
    size_t file_dim, file_num_points, file_aligned_dim;
    load_aligned_bin<T>(std::string(filename), sync_data, file_num_points,
                        file_dim, file_aligned_dim);

    if (file_dim != _dim) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << _dim << " dimension,"
             << "but file has " << file_dim << " dimension." << std::endl;
      std::cerr << stream.str() << std::endl;
      free(sync_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    if (num_points_to_load > _max_points ||
        num_points_to_load > file_num_points) {
      std::stringstream stream;
      stream << "ERROR: Requesting to load " << file_num_points << " points,"
             << "but max_points is set to " << _max_points
             << " points, and file has " << file_num_points << " points."
             << std::endl;
      std::cerr << stream.str() << std::endl;
      free(sync_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    _num_points = num_points_to_load;

    size_t num_points_shard = _num_points / _num_shards;
    size_t offset = 0;
    for (size_t i = 0; i < _num_shards; i++) {
      std::vector<TagT> long_tags;
      size_t            long_size =
          (i == _num_shards - 1)
              ? _num_points - (_num_shards - 1) * num_points_shard
              : num_points_shard;
      if (long_size > 0) {
        long_tags.insert(long_tags.begin(), tags.begin() + offset,
                         tags.begin() + offset + long_size);
        offset += long_size;
      }
      _shards[i]->build(sync_data + _aligned_dim * i * num_points_shard,
                        long_tags);
      std::cout << "Built static index on shard " << i << std::endl;
    }
    aligned_free(sync_data);
    MallocExtension::instance()->ReleaseFreeMemory();
  }

  template<typename T, typename TagT>
  void SyncIndex<T, TagT>::search_sync(const T *query, const size_t K,
                                       const unsigned L, TagT *tags,
                                       float *distances) {
    std::vector<Neighbor_Tag<TagT>> best_all_shards;
    std::vector<Neighbor_Tag<TagT>> best_per_shard;
    for (size_t i = 0; i < _num_shards; i++) {
      best_per_shard.clear();
      _shards[i]->search_sync(query, L, L, best_per_shard);
      best_all_shards.insert(best_all_shards.end(), best_per_shard.begin(),
                             best_per_shard.end());
    }
    std::sort(best_all_shards.begin(), best_all_shards.end());

    {
      std::shared_lock<std::shared_timed_mutex> read_lock(_delete_lock);
      size_t                                    pos = 0;
      for (auto iter : best_all_shards) {
        if (_deletion_set.find(iter.tag) == _deletion_set.end()) {
          tags[pos] = iter.tag;
          distances[pos] = iter.dist;
          pos++;
        }
        if (pos == K)
          break;
      }
    }
  }

  template<typename T, typename TagT>
  void SyncIndex<T, TagT>::search_async(const T *query, const size_t K,
                                        const unsigned L, TagT *tags,
                                        float *distances) {
    std::vector<std::future<void>> futures(_num_shards);

    std::vector<Neighbor_Tag<TagT>>              best_all_shards;
    std::vector<std::vector<Neighbor_Tag<TagT>>> best_per_shard(_num_shards);
    for (size_t i = 0; i < _num_shards; i++)
      futures[i] = _shards[i]->search_async(query, L, L, best_per_shard[i]);

    for (size_t i = 0; i < _num_shards; i++) {
      futures[i].wait();
      best_all_shards.insert(best_all_shards.end(), best_per_shard[i].begin(),
                             best_per_shard[i].end());
    }
    std::sort(best_all_shards.begin(), best_all_shards.end());

    /*    _u32 counter=0;
        std::cout<<"pre-aggregation :: ";
        for (auto &p : best_all_shards) {
            std::cout <<counter<<": " << p.tag<<"," << p.dist <<" ";
            counter++;
        }
        std::cout<<std::endl;
    */

    {
      std::shared_lock<std::shared_timed_mutex> read_lock(_delete_lock);
      size_t                                    pos = 0;
      for (auto iter : best_all_shards) {
        if (_deletion_set.find(iter.tag) == _deletion_set.end()) {
          tags[pos] = iter.tag;
          distances[pos] = iter.dist;
          pos++;
        }
        if (pos == K)
          break;
      }
    }
  }

  template<typename T, typename TagT>
  int SyncIndex<T, TagT>::insert(const T *point, const TagT &tag) {
    /*      {
         std::unique_lock<std::shared_timed_mutex> lock(_delete_lock);
         if(_deletion_set.find(tag) != _deletion_set.end())
         {
             _deletion_set.erase(tag);
             return 0;
         }

         }  */
    auto shard = assign_shard();
    if (shard < 0) {
      std::cerr << "Sync index capacity reached, failed to insert."
                << std::endl;
      return -1;
    }

    //    std::unique_lock<std::shared_timed_mutex> lock(_insert_lock);
    if (_shards[shard]->insert(point, tag) != 0) {
      std::cerr << "Shard did not have enough capacity to insert this point"
                << std::endl;
      return -2;
    } else {
      std::unique_lock<std::shared_timed_mutex> lock(_change_lock);
      _num_points++;
    }
    return 0;
  }

  template<typename T, typename TagT>
  int SyncIndex<T, TagT>::assign_shard() {
    std::shared_lock<std::shared_timed_mutex> lock(_insert_lock);
    if (_num_points >= _max_points) {
      std::cerr << "All shards are full, cannot insert more points"
                << ", _num_points = " << _num_points
                << ", _max_points = " << _max_points << std::endl;
      return -1;
    }
    return rand() % _num_shards;
  }

  template<typename T, typename TagT>
  int SyncIndex<T, TagT>::lazy_delete(const TagT &tag) {
    {
      std::unique_lock<std::shared_timed_mutex> lock(_delete_lock);
      _deletion_set.insert(tag);
    }
    {
      std::unique_lock<std::shared_timed_mutex> lock(_change_lock);
      _num_points--;  // Ravi: should this be here, or done at merge?
    }
    return 0;
  }

  template<typename T, typename TagT>
  int SyncIndex<T, TagT>::merge_all(const std::string &save_path) {
    for (size_t i = 0; i < _num_shards; i++) {
      diskann::Timer timer;
      {
        std::unique_lock<std::shared_timed_mutex> lock(_delete_lock);
        _shards[i]->merge_shard(_deletion_set);
      }

      std::cout << "Merge done on shard " << i << " in "
                << timer.elapsed() / 1000 << "ms" << std::endl;
    }

    return 0;
  }

  template<typename T, typename TagT>
  _u64 SyncIndex<T, TagT>::return_nd() {
    return _num_points;
  }

  template<typename T, typename TagT>
  int SyncIndex<T, TagT>::merge_background(Parameters &paras) {
    return 0;
  }

  template<typename T, typename TagT>
  size_t SyncIndex<T, TagT>::load(const std::string &file_prefix) {
    size_t num_points_loaded = 0;
    for (size_t i = 0; i < _num_shards; i++) {
      std::string shard_file_prefix = file_prefix + std::to_string(i);
      num_points_loaded += _shards[i]->load(shard_file_prefix);
    }
    if (num_points_loaded > _max_points) {
      // throw exception and exit here
    }
    _num_points = num_points_loaded;
    return num_points_loaded;
  }

  template<typename T, typename TagT>
  void SyncIndex<T, TagT>::get_active_tags(tsl::robin_set<TagT> &active_tags) {
    active_tags.clear();
    for (size_t i = 0; i < _num_shards; i++) {
      _shards[i]->return_active_tags(active_tags);
    }
  }
  template DISKANN_DLLEXPORT class SyncIndex<float, int>;
  template DISKANN_DLLEXPORT class SyncIndex<float, unsigned>;
  template DISKANN_DLLEXPORT class SyncIndex<_s8, int>;
  template DISKANN_DLLEXPORT class SyncIndex<_s8, unsigned>;
  template DISKANN_DLLEXPORT class SyncIndex<_u8, int>;
  template DISKANN_DLLEXPORT class SyncIndex<_u8, unsigned>;
}  // namespace diskann
