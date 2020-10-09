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
#include "tsl/robin_set.h"
#include "tcmalloc/malloc_extension.h"

//#include <boost/dynamic_bitset.hpp>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include "aux_utils.h"
#include "exceptions.h"
#include "index.h"
#include "pq_flash_index.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "parameters.h"
#include "partition_and_pq.h"
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

  template<typename T, typename TagT>
  Shard<T, TagT>::Shard(Parameters& parameters, size_t max_points, size_t dim,
                        unsigned           num_search_threads,
                        const std::string& save_file_path)
      : _max_points(max_points), _dim(dim),
        _num_search_threads(num_search_threads), _temp_index(nullptr),
        _active(true), _clearing_short_term(false) {
    _short_term_max_points = SHORT_MAX_POINTS;
    _short_term_index = std::make_shared<diskann::Index<T, TagT>>(
        diskann::L2, dim, _short_term_max_points, 1, 1, 1, 0);

    _paras_mem.Set<unsigned>("L", parameters.Get<unsigned>("L_mem"));
    _paras_mem.Set<unsigned>("R", parameters.Get<unsigned>("R_mem"));
    _paras_mem.Set<unsigned>("C", parameters.Get<unsigned>("C"));
    _paras_mem.Set<float>("alpha", parameters.Get<float>("alpha_mem"));
    _paras_mem.Set<unsigned>("num_rnds", 2);
    _paras_mem.Set<bool>("saturate_graph", 0);

    _paras_disk.Set<unsigned>("L", parameters.Get<unsigned>("L_disk"));
    _paras_disk.Set<unsigned>("R", parameters.Get<unsigned>("R_disk"));
    _paras_disk.Set<unsigned>("C", parameters.Get<unsigned>("C"));
    _paras_disk.Set<float>("alpha", parameters.Get<float>("alpha_disk"));
    _paras_disk.Set<unsigned>("num_rnds", 2);
    _paras_disk.Set<bool>("saturate_graph", 0);

    _num_pq_chunks = parameters.Get<_u32>("num_pq_chunks");
    _beamwidth = parameters.Get<_u32>("beamwidth");
    _num_nodes_to_cache = parameters.Get<_u32>("nodes_to_cache");

    _search_tpool = new ThreadPool(num_search_threads);

    _shard_file_path = save_file_path;
    _temp_index_path = _shard_file_path + "_lti_mem.index";
    _disk_index_path = _shard_file_path + "_lti_disk.index";
    _full_precision_path = _shard_file_path + "_lti_full_precision.data";
    _pq_path = _shard_file_path + "_lti_pq";
    _compressed_path = _shard_file_path + "_lti_pq_compressed.bin";

#ifdef _WINDOWS
#ifndef USE_BING_INFRA
    reader.reset(new WindowsAlignedFileReader());
#else
    reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
    reader.reset(new LinuxAlignedFileReader());
#endif

    _disk_index = new diskann::PQFlashIndex<T>(reader);
  }

  template<typename T, typename TagT>
  int Shard<T, TagT>::insert(const T* point, const TagT& tag) {
    if (_active.load() == false) {
      std::cout << "Shard undergoing merge, can not write to _short_term_index "
                << std::endl;
      return -1;
    }

    if ((_short_term_index->get_num_points() + _lti_num_points <
         _max_points - 1000) &&
        (_short_term_index->get_num_points() <
         _short_term_index->return_max_points())) {
      _short_term_index->insert_point(point, _paras_mem, tag);
      return 0;
    }
    return -2;
  }

  template<typename T, typename TagT>
  Shard<T, TagT>::~Shard() {
    // put in destructor code
  }

  template<typename T, typename TagT>
  void Shard<T, TagT>::build(T* source_data,
                             const std::vector<TagT>& long_tags) {
    std::unique_lock<std::shared_timed_mutex> lock(_shard_lock);
    _temp_index = std::make_shared<diskann::Index<T, TagT>>(
        diskann::L2, _dim, _max_points, 1, 1, 1, 0);
    _lti_num_points = long_tags.size();
    _temp_index->build(source_data, long_tags.size(), _paras_disk, long_tags);

    save();

    diskann::create_disk_layout<T>(_full_precision_path, _temp_index_path,
                                   _disk_index_path);

    lti_location_to_tag.clear();
    _temp_index->get_location_to_tag(lti_location_to_tag);

    size_t                   train_size, train_dim;
    std::unique_ptr<float[]> train_data;  //
    //=
    //    std::unique_ptr<float[]>(new float[train_size * train_dim]);

    double p_val = ((double) TRAINING_SET_SIZE_SMALL /
                    (_temp_index->get_num_points() + 1));

    if (_temp_index != nullptr) {
      _temp_index.reset();
      _temp_index = nullptr;
    }
    // generates random sample and sets it to train_data and updates train_size
    gen_random_slice<T>(_full_precision_path, p_val, train_data, train_size,
                        train_dim);

    //    std::cout << "Training data loaded of size " << train_size <<
    //    std::endl;

    std::string pq_pivots_path = _pq_path + "_pivots.bin";
    std::string pq_compressed_vectors_path = _compressed_path;

    //    /*
    generate_pq_pivots(train_data, train_size, (uint32_t) _dim, 256,
                       (uint32_t) _num_pq_chunks, 12, pq_pivots_path);
    //    */
    generate_pq_data_from_pivots<T>(_full_precision_path, 256,
                                    (uint32_t) _num_pq_chunks, pq_pivots_path,
                                    pq_compressed_vectors_path);

    _disk_index->load(_num_search_threads, _pq_path.c_str(),
                      _disk_index_path.c_str());

    // cache bfs levels
    std::vector<uint32_t> node_list;
    //    std::cout << "Caching " << _num_nodes_to_cache
    //              << " BFS nodes around medoid(s)" << std::endl;
    _disk_index->cache_bfs_levels(_num_nodes_to_cache, node_list);
    //  _pFlashIndex->generate_cache_list_from_sample_queries(
    //      warmup_query_file, 15, 6, num_nodes_to_cache, node_list);
    _disk_index->load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();

    MallocExtension::instance()->ReleaseFreeMemory();
  }

  template<typename T, typename TagT>
  void Shard<T, TagT>::save() {
    if (_temp_index != nullptr) {
      _temp_index->save(_temp_index_path.c_str());
      std::string full_precision = _shard_file_path + "_lti_full_precision";
      _temp_index->save_data(full_precision.c_str(), false);
    }
  }

  template<typename T, typename TagT>
  _u64 Shard<T, TagT>::size() {
    std::shared_lock<std::shared_timed_mutex> lock(_shard_lock);
    return _lti_num_points + _short_term_index->get_num_points();
  }

  template<typename T, typename TagT>
  size_t Shard<T, TagT>::load(const std::string& file_prefix) {
    std::string tag_file = _temp_index_path + std::string(".tags");

    _temp_index = std::make_shared<diskann::Index<T, TagT>>(
        diskann::L2, _dim, _max_points, 1, 1, 1, 0);

    _lti_num_points = _temp_index->load_tag(tag_file);
    lti_location_to_tag.clear();
    _temp_index->get_location_to_tag(lti_location_to_tag);

    //    if (_temp_index != nullptr) {
    _temp_index.reset();
    _temp_index = nullptr;
    //    }

    _disk_index->load(_num_search_threads, _pq_path.c_str(),
                      _disk_index_path.c_str());

    // cache bfs levels
    std::vector<uint32_t> node_list;
    _disk_index->cache_bfs_levels(_num_nodes_to_cache, node_list);
    _disk_index->load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();

    return _lti_num_points;
  }

  template<typename T, typename TagT>
  void Shard<T, TagT>::return_active_tags(tsl::robin_set<TagT>& active_tags) {
    for (auto iter : lti_location_to_tag)
      active_tags.insert(iter.second);
  }

  template<typename T, typename TagT>
  int Shard<T, TagT>::check_size() {
    if ((_short_term_index->get_num_points() + _lti_num_points <
         _max_points - 1000) &&
        (_short_term_index->get_num_points() <
         _short_term_index->return_max_points()))
      return 0;
    return -1;
  }

  template<typename T, typename TagT>
  void Shard<T, TagT>::search_sync(const T* query, const size_t K,
                                   const size_t                     L,
                                   std::vector<Neighbor_Tag<TagT>>& best) {
    assert(best.size() == 0);
    std::vector<Neighbor_Tag<TagT>> short_search_result;
    {
      size_t                                    searchK = K + 1;
      std::shared_lock<std::shared_timed_mutex> lock(_shard_lock);
      std::vector<_u64>                         query_result_ids_64(searchK);
      std::vector<float>                        query_result_dists(searchK);
      //     _temp_index->search(query, K, L, best);

      _disk_index->cached_beam_search(
          query, searchK, L, query_result_ids_64.data(),
          query_result_dists.data(), _beamwidth, nullptr);
      for (_u32 i = 0; i < K + 1; i++) {
        if (lti_location_to_tag.find((_u32) query_result_ids_64[i]) !=
            lti_location_to_tag.end()) {
          best.emplace_back(lti_location_to_tag[(_u32) query_result_ids_64[i]],
                            query_result_dists[i]);
        }
      }
    }
    if ((!_clearing_short_term) && (_short_term_index->get_num_points() > 0)) {
      _short_term_index->search(query, K, (_u32) L, short_search_result);
      best.insert(best.end(), short_search_result.begin(),
                  short_search_result.end());
      std::sort(best.begin(), best.end());
    }

    if (best.size() > K)
      best.erase(best.begin() + K, best.end());
  }

  template<typename T, typename TagT>
  std::future<void> Shard<T, TagT>::search_async(
      const T* query, const size_t K, const size_t L,
      std::vector<Neighbor_Tag<TagT>>& best) {
    return _search_tpool->enqueue(&Shard<T, TagT>::search_sync, this, query, K,
                                  L, std::ref(best));
  }

  template<typename T, typename TagT>
  void Shard<T, TagT>::update_pq_pivots(std::string old_pivots_file,
                                        std::string new_pivots_file) {
  }

  template<typename T, typename TagT>
  void Shard<T, TagT>::update_compressed_vectors(
      std::string old_compressed_file, std::string incr_compressed_file,
      std::unordered_map<TagT, unsigned>&          incr_tag_to_loc,
      std::unordered_map<unsigned, TagT>&          new_loc_to_tag,
      std::string new_compressed_file) {
    std::unique_ptr<_u8[]> old_data;
    std::unique_ptr<_u8[]> incr_data;
    std::unique_ptr<_u8[]> new_data;

    _u64 old_num_points, old_num_chunks;
    _u64 incr_num_points, incr_num_chunks;
    diskann::load_bin<_u8>(old_compressed_file, old_data, old_num_points,
                           old_num_chunks);
    diskann::load_bin<_u8>(incr_compressed_file, incr_data, incr_num_points,
                           incr_num_chunks);

    /*    std::string old_compressed_full = old_compressed_file + "_full.bin";
        std::string incr_compressed_full = incr_compressed_file + "_full.bin";
        std::string new_compressed_full = new_compressed_file + "_full.bin";
        std::unique_ptr<float[]> old_full_pq;
        std::unique_ptr<float[]> incr_full_pq;
        std::unique_ptr<float[]> new_full_pq;

        _u64 old_num_pq_pts, old_dim, incr_dim, incr_num_pq_pts;
        diskann::load_bin<float>(old_compressed_full, old_full_pq,
       old_num_pq_pts,
                                 old_dim);
        diskann::load_bin<float>(incr_compressed_full, incr_full_pq,
                                 incr_num_pq_pts, incr_dim);

        if ((old_num_pq_pts != old_num_points) ||
            (incr_num_pq_pts != incr_num_points)) {
          std::cout << "Number of points does not match." << std::endl;
          exit(-1);
        }
        */

    if (old_num_points != lti_location_to_tag.size() + 1) {
      std::cout
          << "Error in old file, mis-match w.r.t tag_to_location structure"
          << std::endl;
      exit(-1);
    }

    std::unordered_map<TagT, unsigned> old_tag_to_loc;
    for (auto iter : lti_location_to_tag) {
      old_tag_to_loc[iter.second] = iter.first;
    }

    _u64 num_chunks = old_num_chunks;
    _u64 new_num_points = new_loc_to_tag.size() + 1;

    new_data = std::unique_ptr<_u8[]>(new _u8[new_num_points * num_chunks]);
    /*    new_full_pq = std::unique_ptr<float[]>(new float[new_num_points *
     * _dim]); */
    _u64 count_old = 0, count_incr = 0;

    // copying frozen point's centroid information
    std::memcpy(new_data.get() + (new_num_points - 1) * num_chunks,
                old_data.get() + (old_num_points - 1) * num_chunks,
                num_chunks * sizeof(_u8));

    /*    std::memcpy(new_full_pq.get() + (new_num_points - 1) * _dim,
                    old_full_pq.get() + (old_num_pq_pts - 1) * _dim,
                    _dim * sizeof(float));  */

    for (unsigned new_loc = 0; new_loc + 1 < new_num_points; new_loc++) {
      auto new_tag = new_loc_to_tag[new_loc];
      if (old_tag_to_loc.find(new_tag) != old_tag_to_loc.end()) {
        count_old++;
        auto old_loc = old_tag_to_loc[new_tag];
        std::memcpy(new_data.get() + ((_u64) new_loc) * num_chunks,
                    old_data.get() + ((_u64) old_loc) * num_chunks,
                    num_chunks * sizeof(_u8));
        /*        std::memcpy(new_full_pq.get() + ((_u64) new_loc) * _dim,
                            old_full_pq.get() + ((_u64) old_loc) * _dim,
                            _dim * sizeof(float));  */

      } else if (incr_tag_to_loc.find(new_tag) != incr_tag_to_loc.end()) {
        count_incr++;
        auto incr_loc = incr_tag_to_loc[new_tag];
        std::memcpy(new_data.get() + ((_u64) new_loc) * num_chunks,
                    incr_data.get() + ((_u64) incr_loc) * num_chunks,
                    num_chunks * sizeof(_u8));
        /*        std::memcpy(new_full_pq.get() + ((_u64) new_loc) * _dim,
                            incr_full_pq.get() + ((_u64) incr_loc) * _dim,
                            _dim * sizeof(float));  */

      } else {
        std::cout << "Error. new tag found neither in old LTI nor in "
                     "incremental points"
                  << std::endl;
      }
    }

    diskann::save_bin<_u8>(new_compressed_file, new_data.get(), new_num_points,
                           num_chunks);
    /*    diskann::save_bin<float>(new_compressed_full, new_full_pq.get(),
                                 new_num_points, _dim); */
  }

  template<typename T, typename TagT>
  int Shard<T, TagT>::merge_shard(tsl::robin_set<TagT>& deletion_set) {
    malloc_stats();

    bool expected_active = true;
    if (_active.compare_exchange_strong(expected_active, false)) {
      std::cout << "Starting merge on shard " << std::endl;
    } else {
      std::cout << "Unable to start merge" << std::endl;
      return -1;
    }

    if (_temp_index != nullptr) {
      _temp_index.reset();
      _temp_index = nullptr;
    }

    auto data_file = _temp_index_path + ".data";
    _temp_index = std::make_shared<diskann::Index<T, TagT>>(
        diskann::L2, _dim, _max_points, 1, 1, 1, 0);

    _temp_index->load(_temp_index_path.c_str(), data_file.c_str(), true);

    std::vector<TagT> failed_tags;
    if (deletion_set.size() > 0) {
      _temp_index->enable_delete();
      _temp_index->lazy_delete(deletion_set, failed_tags);
      diskann::Timer timer;
      _temp_index->disable_delete(_paras_disk, true);
      std::cout << "Deletion time  : " << timer.elapsed() / 1000 << "ms"
                << std::endl;
    }

    _u64 aligned_dim = ROUND_UP(_dim, 8);
    std::unordered_map<TagT, unsigned> tag_to_loc_short;
    std::string incr_compressed_vectors_path;

    if (_short_term_index->get_num_points() > 0) {
      std::unique_ptr<T[]> ret_data = std::unique_ptr<T[]>(
          new T[_short_term_index->get_num_points() * aligned_dim]);

      if (_short_term_index->extract_data(ret_data.get(), tag_to_loc_short) !=
          0) {
        std::cout << "Could not access short_term_index data. Exiting...."
                  << std::endl;
        exit(-1);
      }

      std::string temp_incr_bin = _shard_file_path + "_temp_incr.bin";
      save_bin<T>(temp_incr_bin, ret_data.get(),
                  _short_term_index->get_num_points(), aligned_dim);

      std::string pq_pivots_path = _pq_path + "_pivots.bin";
      incr_compressed_vectors_path =
          _shard_file_path + "_incr_pq_compressed.bin";
      generate_pq_data_from_pivots<T>(_shard_file_path + "_temp_incr.bin", 256,
                                      (uint32_t) _num_pq_chunks, pq_pivots_path,
                                      incr_compressed_vectors_path);
      std::vector<unsigned> new_tags;
      std::vector<unsigned> new_loc;

      diskann::Timer timer;
      for (auto iter : tag_to_loc_short) {
        if (deletion_set.find(iter.first) == deletion_set.end()) {
          new_tags.push_back(iter.first);
          new_loc.push_back(iter.second);
        }
      }
      std::unique_ptr<T[]> active_data =
          std::unique_ptr<T[]>(new T[new_tags.size() * aligned_dim]);
#pragma omp parallel for
      for (_s64 i = 0; i < (_s64) new_loc.size(); i++) {
        std::memcpy(active_data.get() + i * aligned_dim,
                    ret_data.get() + new_loc[i] * aligned_dim,
                    aligned_dim * sizeof(T));
      }

      _temp_index->compact_data_for_insert();
      timer.reset();
#pragma omp parallel for schedule(dynamic, 128)
      for (_s64 i = 0; i < (_s64) new_tags.size(); i++) {
        _temp_index->insert_point(active_data.get() + i * _dim, _paras_disk,
                                  new_tags[i]);
      }
      std::cout << "Insertion time : " << timer.elapsed() / 1000 << "msec"
                << std::endl;
      std::cout << "Running sync_prune for all points to bound degree"
                << std::endl;
      _temp_index->prune_all_nbrs(_paras_disk);
    }
    deletion_set.clear();
    for (auto tag : failed_tags) {
      if (_short_term_index->get_num_points() > 0) {
        if (tag_to_loc_short.find(tag) == tag_to_loc_short.end())
          deletion_set.insert(tag);
      } else
        deletion_set.insert(tag);
    }

    {
      auto save_temp = _temp_index_path;
      _temp_index->save(save_temp.c_str());

      _lti_num_points = _temp_index->get_num_points();

      std::unordered_map<unsigned, TagT> _temp_index_loc_to_tag;
      _temp_index->get_location_to_tag(_temp_index_loc_to_tag);
      std::string full_precision = _shard_file_path + "_lti_full_precision";
      _temp_index->save_data(full_precision.c_str(), false);

      {
        std::unique_lock<std::shared_timed_mutex> lock(
            _shard_lock);  // disk index will not cater search threads
        if (_short_term_index->get_num_points() > 0) {
          update_compressed_vectors(
              _compressed_path, incr_compressed_vectors_path, tag_to_loc_short,
              _temp_index_loc_to_tag, _compressed_path);
        }

        diskann::create_disk_layout<T>(_full_precision_path, _temp_index_path,
                                       _disk_index_path);

        lti_location_to_tag.clear();
        _temp_index->get_location_to_tag(lti_location_to_tag);
      }
      //      _temp_index->check_graph_quality(20000, _paras_disk);
      // update PQ table
      // create_disk_layout
      if (_temp_index != nullptr) {
        _temp_index.reset();
        _temp_index = nullptr;
      }
      delete (_disk_index);

      _disk_index = new diskann::PQFlashIndex<T>(reader);
      _disk_index->load(_num_search_threads, _pq_path.c_str(),
                        _disk_index_path.c_str());

      // cache bfs levels
      std::vector<uint32_t> node_list;
      //      std::cout << "Caching " << _num_nodes_to_cache
      //                << " BFS nodes around medoid(s)" << std::endl;
      _disk_index->cache_bfs_levels(_num_nodes_to_cache, node_list);
      //  _pFlashIndex->generate_cache_list_from_sample_queries(
      //      warmup_query_file, 15, 6, num_nodes_to_cache, node_list);
      _disk_index->load_cache_list(node_list);
      node_list.clear();
      node_list.shrink_to_fit();
    }

    bool expected_clearing = false;
    expected_active = false;
    _clearing_short_term.compare_exchange_strong(expected_clearing, true);
    _short_term_index.reset();
    _short_term_index = nullptr;
    _short_term_index = std::make_shared<diskann::Index<T, TagT>>(
        diskann::L2, _dim, _short_term_max_points, 1, 1, 1, 0);
    expected_clearing = true;
    assert(expected_clearing == true);
    _clearing_short_term.compare_exchange_strong(expected_clearing, false);
    assert(expected_active == false);
    _active.compare_exchange_strong(expected_active, true);

    MallocExtension::instance()->ReleaseFreeMemory();
    //    malloc_stats();

    return 0;
  }

  template DISKANN_DLLEXPORT class Shard<float, int>;
  template DISKANN_DLLEXPORT class Shard<float, unsigned>;

  template DISKANN_DLLEXPORT class Shard<int8_t, int>;
  template DISKANN_DLLEXPORT class Shard<int8_t, unsigned>;

  template DISKANN_DLLEXPORT class Shard<uint8_t, int>;
  template DISKANN_DLLEXPORT class Shard<uint8_t, unsigned>;
}
