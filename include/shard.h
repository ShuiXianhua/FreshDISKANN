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
#include "pq_flash_index.h"
#include "parameters.h"
#include "Neighbor_Tag.h"
#include "threadpool.h"

#define SHORT_MAX_POINTS 1100000

namespace diskann {

  template<typename T, typename TagT = int>
  class Shard {
   public:
    // constructor
    Shard(Parameters& parameters, size_t max_points, size_t dim,
          unsigned num_search_threads, const std::string& save_file_path);

    ~Shard();

    void build(T* source_data, const std::vector<TagT>& long_tags);
    void save();

    // call merge on a shard
    int merge_shard(tsl::robin_set<TagT>& deletion_set);

    // insertion function - insert into short_term_index
    int insert(const T* point, const TagT& tag);

    // gives go-ahead for processing insertion into short_term_index, does not
    // guarantee insertion, need another check in the insertion function
    int check_size();

    _u64 size();

    // search function - search both short_term_index and long_term_index and
    // return with top L candidate tags of the shard
    void search_sync(const T* query, const size_t K, const size_t L,
                     std::vector<Neighbor_Tag<TagT>>& best);

    std::future<void> search_async(const T* query, const size_t K,
                                   const size_t                     L,
                                   std::vector<Neighbor_Tag<TagT>>& best);

    size_t load(const std::string& file_prefix);

    void return_active_tags(tsl::robin_set<TagT>& active_tags);

   private:
    void update_compressed_vectors(
        std::string old_compressed_file, std::string incr_compressed_file,
        std::unordered_map<TagT, unsigned>&          incr_tag_to_loc,
        std::unordered_map<unsigned, TagT>&          new_loc_to_tag,
        std::string new_compressed_file);

    void update_pq_pivots(std::string old_pivots_file,
                          std::string new_pivots_file);

    size_t _max_points;
    size_t _short_term_max_points = 0;
    size_t _lti_num_points = 0;
    size_t _dim;
    _u32   _num_pq_chunks;
    _u32   _num_nodes_to_cache;
    float  _pq_sampling_rate;
    _u32   _num_search_threads;
    _u32   _beamwidth;

    // in memory index for merge
    PQFlashIndex<T>*                   _disk_index;
    std::shared_ptr<AlignedFileReader> reader = nullptr;
    std::unordered_map<unsigned, TagT> lti_location_to_tag;

    std::shared_ptr<Index<T, TagT>> _short_term_index = nullptr;
    std::shared_ptr<Index<T, TagT>> _temp_index = nullptr;

    /*    Index<T, TagT>* _short_term_index;
        Index<T, TagT>* _temp_index;    */

    diskann::Parameters _paras_mem;
    diskann::Parameters _paras_disk;

    std::atomic_bool _active;  // true except when merging
    std::atomic_bool
                            _clearing_short_term;  // don't search short_term_index if true
    std::shared_timed_mutex _shard_lock;  // mutex to switch long_term_index and
                                          // temp_index pointers

    ThreadPool* _search_tpool;

    std::string _shard_file_path;
    std::string _temp_index_path;
    std::string _disk_index_path;
    std::string _full_precision_path;
    std::string _pq_path;
    std::string _compressed_path;
  };
}
