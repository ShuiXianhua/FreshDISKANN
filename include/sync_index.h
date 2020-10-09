#pragma once

#include <cassert>
#include <map>
#include <shared_mutex>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <utility>
#include "tsl/robin_set.h"

#include "index.h"
#include "shard.h"
#include "Neighbor_Tag.h"
#include "parameters.h"

#define SHARD_SLACK_FACTOR 1.1

namespace diskann {
  template<typename T, typename TagT>
  class SyncIndex {
   public:
    // constructor function - will distribute points among shard, also call
    // constructor for all shards
    DISKANN_DLLEXPORT SyncIndex(const size_t max_points, const size_t dim,
                                const size_t num_shards, Parameters& parameters,
                                const unsigned     num_search_threads_per_shard,
                                const std::string& save_file_path);

    // destructor - will free up space from _sync_data
    DISKANN_DLLEXPORT ~SyncIndex();

    // allocates data to shards, generates frozens point for short and
    // long_term_index
    // builds long terms index in each shard, saves data of every shard on a
    // separate file
    DISKANN_DLLEXPORT void build(const char*              file_name,
                                 const size_t             num_points_to_load,
                                 const std::vector<TagT>& tags);

    // Insert new point in to one of the shards
    DISKANN_DLLEXPORT int insert(const T* point, const TagT& tag);
    // log delete requests in global delete set
    DISKANN_DLLEXPORT int lazy_delete(const TagT& tag);

    // synchronoulsy calls search on each shard, aggregate and re-rank results
    DISKANN_DLLEXPORT void search_sync(const T* query, const size_t K,
                                       const unsigned L, TagT* tags,
                                       float* distances);

    // synchronoulsy calls search on each shard, aggregate and re-rank results
    DISKANN_DLLEXPORT void search_async(const T* query, const size_t K,
                                        const unsigned L, TagT* tags,
                                        float* distances);

    DISKANN_DLLEXPORT int merge_all(const std::string& save_path);

    DISKANN_DLLEXPORT int merge_background(Parameters& paras);

    DISKANN_DLLEXPORT _u64 return_nd();

    DISKANN_DLLEXPORT size_t load(const std::string& file_prefix);

    DISKANN_DLLEXPORT void get_active_tags(tsl::robin_set<TagT>& active_tags);

   protected:
    int assign_shard();
    // void shutdown(const char* file_prefix);

   private:
    const size_t _num_shards;
    //    T*           _sync_data = nullptr;  // store coordinates of all the
    //    points
    size_t _num_points;  // number of points present currently in all shards
    size_t _dim;
    size_t _aligned_dim;
    int    _alignment_factor;
    size_t _max_points;  // maximum allowed number of points

    std::vector<Shard<T, TagT>*> _shards;
    tsl::robin_set<TagT>    _deletion_set;  // change to set
    std::shared_timed_mutex _delete_lock;   // lock to access _deletion_set
    std::shared_timed_mutex _insert_lock;   // Lock to check capacity
    std::shared_timed_mutex _change_lock;   // Lock to check capacity
  };
}
