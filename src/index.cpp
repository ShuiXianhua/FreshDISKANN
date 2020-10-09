#include <algorithm>
#include <atomic>
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
//#include <semaphore.h>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <string>
#include "tsl/robin_set.h"
#include "tsl/robin_map.h"
#include <unordered_map>

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include "exceptions.h"
#include "index.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "parameters.h"
#include "partition_and_pq.h"
#include "timer.h"
#include "utils.h"
#include "windows_customizations.h"
#include "tcmalloc/malloc_extension.h"
#ifdef _WINDOWS
#include <xmmintrin.h>
#endif

#include "Neighbor_Tag.h"
// only L2 implemented. Need to implement inner product search
namespace {
  template<typename T>
  diskann::Distance<T> *get_distance_function(diskann::Metric m);

  template<>
  diskann::Distance<float> *get_distance_function(diskann::Metric m) {
    if (m == diskann::Metric::L2)
      return new diskann::DistanceL2();
    else {
      std::stringstream stream;
      stream << "Only L2 metric supported as of now. Email "
                "gopalsr@microsoft.com if you need cosine similarity or inner "
                "product."
             << std::endl;
      std::cerr << stream.str() << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  template<>
  diskann::Distance<int8_t> *get_distance_function(diskann::Metric m) {
    if (m == diskann::Metric::L2)
      return new diskann::DistanceL2Int8();
    else {
      std::stringstream stream;
      stream << "Only L2 metric supported as of now. Email "
                "gopalsr@microsoft.com if you need cosine similarity or inner "
                "product."
             << std::endl;
      std::cerr << stream.str() << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  template<>
  diskann::Distance<uint8_t> *get_distance_function(diskann::Metric m) {
    if (m == diskann::Metric::L2)
      return new diskann::DistanceL2UInt8();
    else {
      std::stringstream stream;
      stream << "Only L2 metric supported as of now. Email "
                "gopalsr@microsoft.com if you need cosine similarity or inner "
                "product."
             << std::endl;
      std::cerr << stream.str() << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }
}  // namespace

namespace diskann {

  // Initialize an index with metric m, load the data of type T with filename
  // (bin), and initialize max_points
  template<typename T, typename TagT>

  Index<T, TagT>::Index(Metric m, const size_t dim, const size_t max_points,
                        const size_t num_frozen_pts, const bool enable_tags,
                        const bool store_data, const bool support_eager_delete)
      : _dim(dim), _max_points(max_points), _num_frozen_pts(num_frozen_pts),
        _enable_tags(enable_tags), _support_eager_delete(support_eager_delete),
        _store_data(store_data) {
    // data is stored to _nd * aligned_dim matrix with necessary
    // zero-padding
    _aligned_dim = ROUND_UP(_dim, 8);
    alloc_aligned(((void **) &_data),
                  (_max_points + _num_frozen_pts) * _aligned_dim * sizeof(T),
                  8 * sizeof(T));
    std::memset(_data, 0,
                (_max_points + _num_frozen_pts) * _aligned_dim * sizeof(T));
    generate_frozen_point();
    _ep = (unsigned) _max_points;
    _final_graph.reserve(_max_points + _num_frozen_pts);
    _final_graph.resize(_max_points + _num_frozen_pts);
    if (_support_eager_delete) {
      _in_graph.reserve(_max_points + _num_frozen_pts);
      _in_graph.resize(_max_points + _num_frozen_pts);
    }

    this->_distance = ::get_distance_function<T>(m);
    _locks = std::vector<std::mutex>(_max_points + _num_frozen_pts);
    if (_support_eager_delete)
      _locks_in = std::vector<std::mutex>(_max_points + _num_frozen_pts);

    _width = 0;
  }

  /*   template<typename T, typename TagT>
    Index<T, TagT>::Index() = default;    */

  template<typename T, typename TagT>
  Index<T, TagT>::Index(Index<T, TagT> *index)
      : _dim(index->_dim), _aligned_dim(index->_aligned_dim), _nd(index->_nd),
        _max_points(index->_max_points),
        _num_frozen_pts(index->_num_frozen_pts), _width(index->_width),
        _ep(index->_ep), _has_built(index->_has_built),
        _saturate_graph(index->_saturate_graph),
        _enable_tags(index->_enable_tags),
        _support_eager_delete(index->_support_eager_delete),
        _store_data(index->_store_data) {
    alloc_aligned(((void **) &_data), _nd * _aligned_dim * sizeof(T),
                  8 * sizeof(T));
    std::memset(_data, 0, _nd * _aligned_dim * sizeof(T));

    T *data_copy = _data;
    _data = (T *) realloc(
        _data, (_max_points + _num_frozen_pts) * _aligned_dim * sizeof(T));
    if (_data == nullptr) {
      std::cout << "Realloc failed in constructor, killing programme"
                << std::endl;
      free(data_copy);
      throw diskann::ANNException("Realloc failed", -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    memcpy(_data, index->_data,
           _aligned_dim * (_max_points + _num_frozen_pts) * sizeof(T));
    _final_graph.reserve(_max_points + _num_frozen_pts);
    _final_graph.resize(_max_points + _num_frozen_pts);
    for (size_t i = 0; i < index->_final_graph.size(); i++) {
      _final_graph[i].reserve(index->_final_graph[i].size());
      _final_graph[i].resize(index->_final_graph[i].size());
      for (size_t j = 0; j < _final_graph[i].size(); j++)
        _final_graph[i] =
            index->_final_graph[i];  // replace with vector assignment
    }

    _tag_to_location = index->_tag_to_location;
    _location_to_tag = index->_location_to_tag;

    this->_distance = ::get_distance_function<T>(diskann::L2);
    _locks = std::vector<std::mutex>(_max_points + _num_frozen_pts);
    _ep = index->_ep;
  }

  template<typename T, typename TagT>
  Index<T, TagT>::~Index() {
    delete this->_distance;
    aligned_free(_data);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::clear_index() {
    memset(_data, 0,
           _aligned_dim * (_max_points + _num_frozen_pts) * sizeof(T));
    _nd = 0;
    for (size_t i = 0; i < _final_graph.size(); i++)
      _final_graph[i].clear();

    _tag_to_location.clear();
    _location_to_tag.clear();
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::save_tag(const std::string &tag_filename) {
    std::ofstream out_tags(tag_filename);
    for (unsigned i = 0; i < _nd; i++) {
      out_tags << _location_to_tag[i] << "\n";
    }
    out_tags.close();
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::save_tag_bin(const std::string &tag_filename) {
    TagT * tag_data = new TagT[_nd];
    size_t pos = 0;
    for (size_t i = 0; i < _nd; i++) {
      tag_data[i] = _location_to_tag[i];
    }
    save_bin<TagT>(tag_filename, tag_data, _nd, 1);
    delete[] tag_data;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::save_data(const char *filename,
                                 bool separate_frozen_pt) {
    if (separate_frozen_pt) {
      if ((_store_data) && (_nd != 0))
        save_unaligned_bin(std::string(filename) + std::string(".data"), _data,
                           _nd, _dim, _aligned_dim);
      if (_num_frozen_pts > 0) {
        save_unaligned_bin(std::string(filename) + std::string(".frozen"),
                           _data + _nd * _aligned_dim, _num_frozen_pts, _dim,
                           _aligned_dim);
      }
    } else {
      if ((_store_data) && (_nd != 0))
        save_unaligned_bin(std::string(filename) + std::string(".data"), _data,
                           _nd + _num_frozen_pts, _dim, _aligned_dim);
    }
  }
  // save the graph index on a file as an adjacency list. For each point,
  // first store the number of neighbors, and then the neighbor list (each as
  // 4 byte unsigned)
  template<typename T, typename TagT>
  void Index<T, TagT>::save(const char *filename, bool separate_frozen_pt,
                            bool tags_bin) {
    _change_lock.lock();
    long long     total_gr_edges = 0;
    size_t        index_size = 0;
    std::ofstream out(std::string(filename), std::ios::binary | std::ios::out);

    if (_support_eager_delete)
      if (_eager_done && (!_data_compacted)) {
        if (_nd < _max_points) {
          assert(_final_graph.size() == _max_points + _num_frozen_pts);
          compact_data();
          update_in_graph();

        } else {
          assert(_final_graph.size() == _max_points + _num_frozen_pts);
          if (_enable_tags) {
            //_change_lock.lock();
            if (_can_delete) {
              std::cerr << "Disable deletes and consolidate "
                           "index before saving."
                        << std::endl;
              throw diskann::ANNException(
                  "Disable deletes and consolidate index before "
                  "saving.",
                  -1, __FUNCSIG__, __FILE__, __LINE__);
            }
          }
        }
        _eager_done = false;
      }
    if ((_lazy_done) && (!_data_compacted)) {
      std::vector<unsigned> new_location;
      compact_data();
      assert(_final_graph.size() == _max_points + _num_frozen_pts);
      if (_enable_tags) {
        if (_can_delete || (!_data_compacted)) {
          std::cout << "Disable deletes and consolidate index before "
                       "saving."
                    << std::endl;
          throw diskann::ANNException(
              "Disable deletes and consolidate index before saving.", -1,
              __FUNCSIG__, __FILE__, __LINE__);
        }
      }
      _lazy_done = false;
    }

    unsigned new_ep = _ep;
    compact_frozen_point();

    _u32 max_degree = 0;
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &_width, sizeof(unsigned));
    //    unsigned new_ep = _ep;
    if (_num_frozen_pts > 0 && _nd < _max_points)
      new_ep = (_u32) _nd;
    out.write((char *) &new_ep, sizeof(unsigned));
    for (unsigned i = 0; i < _nd + _num_frozen_pts; i++) {
      unsigned GK = (unsigned) _final_graph[i].size();
      out.write((char *) &GK, sizeof(unsigned));
      out.write((char *) _final_graph[i].data(), GK * sizeof(unsigned));
      max_degree = _final_graph[i].size() > max_degree
                       ? (_u32) _final_graph[i].size()
                       : max_degree;
      total_gr_edges += GK;
    }
    index_size = out.tellp();
    out.seekp(0, std::ios::beg);
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &max_degree, sizeof(_u32));
    out.close();

    if ((_enable_tags) && (_nd != 0)) {
      if (tags_bin)
        save_tag_bin(std::string(filename) + std::string(".tags"));
      else
        save_tag(std::string(filename) + std::string(".tags"));
    }
    _change_lock.unlock();
    save_data(filename, separate_frozen_pt);
    reposition_frozen_point_to_end();
    if (_support_eager_delete) {
      _data_compacted = true;
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::load_frozen_point(std::string filename) {
    std::string   frozen_file = filename + std::string(".frozen");
    std::ifstream frozen(filename + std::string(".frozen"));
    if (!frozen.is_open()) {
      std::cout << "Frozen point file " << frozen_file << " not found"
                << std::endl;
      return;
    }
    std::unique_ptr<T[]> frozen_data;
    load_aligned_bin(frozen_file, frozen_data, _num_frozen_pts, _dim,
                     _aligned_dim);

    if (_num_frozen_pts > 0) {
      memcpy(_data + _max_points * _aligned_dim, frozen_data.get(),
             _aligned_dim * sizeof(T));
      _ep = (_u32) _max_points;

      _final_graph[_max_points].swap(
          _final_graph[_nd]);  // push the frozen point to the end of the index
      for (size_t i = 0; i < _nd; i++) {
        for (size_t j = 0; j < _final_graph[i].size(); j++) {
          if (_final_graph[i][j] == _nd)
            _final_graph[i][j] = (_u32) _max_points;
        }
      }
      std::cout << _max_points << std::endl;
      std::cout << "Frozen point loaded, _ep is now " << _ep << std::endl;
    }
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_tag_bin(const std::string &tag_filename) {
    if (!file_exists(tag_filename)) {
      std::cerr << "Tag file provided does not exist!!! Exiting...."
                << std::endl;
      exit(-1);
    }

    size_t file_dim, file_num_points;
    TagT * tag_data;
    load_bin<TagT>(std::string(tag_filename), tag_data, file_num_points,
                   file_dim);

    if (file_dim != 1) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << _dim << " dimension,"
             << "but file has " << file_dim << " dimension." << std::endl;
      std::cerr << stream.str() << std::endl;
      free(tag_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    /*
    if (file_num_points != _nd) {
      std::stringstream stream;
      stream << "ERROR: Requesting to load " << file_num_points << " points,"
             << "but _nd is set to " << _nd << " points, and file has "
             << file_num_points << " points." << std::endl;
      std::cerr << stream.str() << std::endl;
      free(tag_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
*/
    size_t i = 0;
    for (i = 0; i < file_num_points; i++) {
      TagT tag = *(tag_data + i);
      _location_to_tag[i] = tag;
      _tag_to_location[tag] = i;
    }
    std::cout << "Tags loaded." << std::endl;
    delete[] tag_data;
    return i;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_tag(const std::string &tag_file_name) {
    std::ifstream tag_file;
    tag_file = std::ifstream(tag_file_name);
    if (!tag_file.is_open()) {
      std::cerr << "Tag file not found." << std::endl;
      return 0;
    }
    unsigned id = 0;
    TagT     tag;
    while (tag_file >> tag) {
      _location_to_tag[id] = tag;
      _tag_to_location[tag] = id++;
    }
    std::cout << "Tags loaded." << std::endl;
    tag_file.close();
    return id;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::load_data(const char *filename) {
    if (filename == nullptr) {
      std::cout << "Starting with an empty index." << std::endl;
      _nd = 0;
    } else

        if (!file_exists(filename)) {
      std::cerr << "Data file provided does not exist!!! Exiting...."
                << std::endl;
      exit(-1);
    }

    size_t file_dim, file_num_points;
    load_aligned_data<T>(std::string(filename), _data, file_num_points,
                         file_dim, _aligned_dim);

    if (file_dim != _dim) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << _dim << " dimension,"
             << "but file has " << file_dim << " dimension." << std::endl;
      std::cerr << stream.str() << std::endl;
      free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    if (file_num_points > _max_points) {
      std::stringstream stream;
      stream << "ERROR: Requesting to load " << file_num_points << " points,"
             << "but max_points is set to " << _max_points
             << " points, and file has " << file_num_points << " points."
             << std::endl;
      std::cerr << stream.str() << std::endl;
      free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    _nd = file_num_points;
  }
  // load the index from file and update the width (max_degree), ep (navigating
  // node id), and _final_graph (adjacency list)

  template<typename T, typename TagT>
  void Index<T, TagT>::load(const char *filename, const char *data_file,
                            const bool load_tags, const char *tag_filename,
                            bool tags_bin) {
    if (!validate_file_size(filename)) {
      return;
    }
    load_data(data_file);
    std::ifstream in(filename, std::ios::binary);
    size_t        expected_file_size;
    in.read((char *) &expected_file_size, sizeof(_u64));
    in.read((char *) &_width, sizeof(unsigned));
    in.read((char *) &_ep, sizeof(unsigned));
    std::cout << "Loading vamana index " << filename << "..." << std::flush;

    size_t   cc = 0;
    unsigned nodes = 0;
    while (!in.eof()) {
      unsigned k;
      in.read((char *) &k, sizeof(unsigned));
      if (in.eof())
        break;
      cc += k;
      ++nodes;
      std::vector<unsigned> tmp(k);
      in.read((char *) tmp.data(), k * sizeof(unsigned));
      _final_graph[nodes - 1].swap(tmp);
      //      if (nodes % 10000000 == 0)
      //        std::cout << "." << std::flush;
    }

    std::cout << "done. Index has " << nodes << " nodes and " << cc
              << " out-edges, _ep is set to " << _ep << std::endl;

    load_frozen_point(filename);

    if (nodes != _nd + _num_frozen_pts) {
      std::cout << "ERROR. mismatch in number of points. Graph has "
                << _final_graph.size() << " points and loaded dataset has "
                << _nd << " points. " << std::endl;
      return;
    }

    if (load_tags) {
      if (_enable_tags == false)
        std::cout << "Enabling tags." << std::endl;
      _enable_tags = true;
      std::string tag_file_name;
      if (tag_filename == NULL)
        tag_file_name = std::string(filename) + std::string(".tags");
      else {
        tag_file_name = std::string(tag_filename);
      }
      _u64 num_tags;
      if (tags_bin)
        num_tags = load_tag_bin(tag_file_name);
      else
        num_tags = load_tag(tag_file_name);
      if (num_tags != _nd) {
        std::cout << "#tags loaded is not equal to _nd. Exiting" << std::endl;
        exit(-1);
      }
    }

    else {
      unsigned id = 0;
      unsigned tag = 0;
      while (id < _nd) {
        _location_to_tag[id] = tag;
        _tag_to_location[tag] = id++;
        tag++;
      }
    }
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::get_vector_by_tag(TagT &tag, T *vec) {
    if (_tag_to_location.find(tag) == _tag_to_location.end()) {
      std::cout << "Tag does not exist" << std::endl;
      return -1;
    }
    unsigned location = _tag_to_location[tag];
    // memory should be allocated for vec before calling this function
    memcpy((void *) vec, (void *) (_data + (size_t)(location * _aligned_dim)),
           (size_t) _aligned_dim * sizeof(T));
    return 0;
  }

  /**************************************************************
   *      Support for Static Index Building and Searching
   **************************************************************/

  /* This function finds out the navigating node, which is the medoid node
   * in the graph.
   */
  template<typename T, typename TagT>
  unsigned Index<T, TagT>::calculate_entry_point() {
    // allocate and init centroid
    float *center = new float[_aligned_dim]();
    for (size_t j = 0; j < _aligned_dim; j++)
      center[j] = 0;

    for (size_t i = 0; i < _nd; i++)
      for (size_t j = 0; j < _aligned_dim; j++)
        center[j] += (float) _data[i * _aligned_dim + j];

    for (size_t j = 0; j < _aligned_dim; j++)
      center[j] /= _nd;

    // compute all to one distance
    float * distances = new float[_nd]();
#pragma omp parallel for schedule(static, 65536)
    for (_s64 i = 0; i < (_s64) _nd; i++) {
      // extract point and distance reference
      float &  dist = distances[i];
      const T *cur_vec = _data + (i * (size_t) _aligned_dim);
      dist = 0;
      float diff = 0;
      for (size_t j = 0; j < _aligned_dim; j++) {
        diff =
            (center[j] - (float) cur_vec[j]) * (center[j] - (float) cur_vec[j]);
        dist += diff;
      }
    }
    // find imin
    unsigned min_idx = 0;
    float    min_dist = distances[0];
    for (unsigned i = 1; i < _nd; i++) {
      if (distances[i] < min_dist) {
        min_idx = i;
        min_dist = distances[i];
      }
    }

    delete[] distances;
    delete[] center;
    return min_idx;
  }

  /* iterate_to_fixed_point():
   * node_coords : point whose neighbors to be found.
   * init_ids : ids of initial search list.
   * Lsize : size of list.
   * beam_width: beam_width when performing indexing
   * expanded_nodes_info: will contain all the node ids and distances from
   * query that are expanded
   * expanded_nodes_ids : will contain all the nodes that are expanded during
   * search.
   * best_L_nodes: ids of closest L nodes in list
   */
  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::iterate_to_fixed_point(
      const T *node_coords, const unsigned Lsize,
      const std::vector<unsigned> &init_ids,
      std::vector<Neighbor> &      expanded_nodes_info,
      tsl::robin_set<unsigned> &   expanded_nodes_ids,
      std::vector<Neighbor> &best_L_nodes, bool ret_frozen) {
    best_L_nodes.resize(Lsize + 1);
    for (unsigned i = 0; i < Lsize + 1; i++) {
      best_L_nodes[i].distance = std::numeric_limits<float>::max();
    }
    expanded_nodes_info.reserve(10 * Lsize);
    expanded_nodes_ids.reserve(10 * Lsize);

    unsigned                 l = 0;
    Neighbor                 nn;
    tsl::robin_set<unsigned> inserted_into_pool;
    inserted_into_pool.reserve(Lsize * 20);

    for (auto id : init_ids) {
      if (id >= _max_points + _num_frozen_pts) {
        std::cout << "Wrong id : " << id << std::endl;
        exit(-1);
      }
      nn = Neighbor(id,
                    _distance->compare(_data + _aligned_dim * (size_t) id,
                                       node_coords, (unsigned) _aligned_dim),
                    true);
      if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
        inserted_into_pool.insert(id);
        best_L_nodes[l++] = nn;
      }
      if (l == Lsize)
        break;
    }

    /* sort best_L_nodes based on distance of each point to node_coords */
    std::sort(best_L_nodes.begin(), best_L_nodes.begin() + l);
    unsigned k = 0;
    uint32_t hops = 0;
    uint32_t cmps = 0;

    while (k < l) {
      unsigned nk = l;

      if (best_L_nodes[k].flag) {
        best_L_nodes[k].flag = false;
        auto n = best_L_nodes[k].id;
        if (!(best_L_nodes[k].id == _ep && _num_frozen_pts > 0 &&
              !ret_frozen)) {
          expanded_nodes_info.emplace_back(best_L_nodes[k]);
          expanded_nodes_ids.insert(n);
        }
        std::vector<unsigned> des;
        {
          LockGuard guard(_locks[n]);
          for (unsigned m = 0; m < _final_graph[n].size(); m++) {
            if (_final_graph[n][m] >= _max_points + _num_frozen_pts) {
              std::cout << "Wrong id : " << _final_graph[n][m]
                        << " at line 663. n = " << n << std::endl;
              exit(-1);
            }
            des.emplace_back(_final_graph[n][m]);
          }
        }

        for (unsigned m = 0; m < des.size(); ++m) {
          unsigned id = des[m];
          if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
            inserted_into_pool.insert(id);

            if ((m + 1) < des.size()) {
              auto nextn = des[m + 1];
              diskann::prefetch_vector(
                  (const char *) _data + _aligned_dim * (size_t) nextn,
                  sizeof(T) * _aligned_dim);
            }

            cmps++;
            float dist = _distance->compare(node_coords,
                                            _data + _aligned_dim * (size_t) id,
                                            (unsigned) _aligned_dim);

            if (dist >= best_L_nodes[l - 1].distance && (l == Lsize))
              continue;

            Neighbor nn(id, dist, true);
            unsigned r = InsertIntoPool(best_L_nodes.data(), l, nn);
            if (l < Lsize)
              ++l;
            if (r < nk)
              nk = r;
          }
        }

        if (nk <= k)
          k = nk;
        else
          ++k;
      } else
        k++;
    }
    return std::make_pair(hops, cmps);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::iterate_to_fixed_point(
      const T *node_coords, const unsigned Lindex,
      std::vector<Neighbor> &expanded_nodes_info,
      tsl::robin_map<uint32_t, T *> &coord_map, bool return_frozen_pt) {
    std::vector<uint32_t> init_ids;
    init_ids.push_back(this->_ep);
    std::vector<Neighbor>    best_L_nodes;
    tsl::robin_set<uint32_t> expanded_nodes_ids;
    this->iterate_to_fixed_point(node_coords, Lindex, init_ids,
                                 expanded_nodes_info, expanded_nodes_ids,
                                 best_L_nodes, return_frozen_pt);
    for (Neighbor &einf : expanded_nodes_info) {
      T *coords =
          this->_data + (uint64_t) einf.id * (uint64_t) this->_aligned_dim;
      coord_map.insert(std::make_pair(einf.id, coords));
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_expanded_nodes(
      const size_t node_id, const unsigned Lindex,
      std::vector<unsigned>     init_ids,
      std::vector<Neighbor> &   expanded_nodes_info,
      tsl::robin_set<unsigned> &expanded_nodes_ids) {
    const T *             node_coords = _data + _aligned_dim * node_id;
    std::vector<Neighbor> best_L_nodes;

    if (init_ids.size() == 0)
      init_ids.emplace_back(_ep);

    iterate_to_fixed_point(node_coords, Lindex, init_ids, expanded_nodes_info,
                           expanded_nodes_ids, best_L_nodes);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::occlude_list(std::vector<Neighbor> &pool,
                                    const unsigned location, const float alpha,
                                    const unsigned degree, const unsigned maxc,
                                    std::vector<Neighbor> &result) {
    auto               pool_size = (_u32) pool.size();
    std::vector<float> occlude_factor(pool_size, 0);
    occlude_list(pool, location, alpha, degree, maxc, result, occlude_factor);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::occlude_list(std::vector<Neighbor> &pool,
                                    const unsigned location, const float alpha,
                                    const unsigned degree, const unsigned maxc,
                                    std::vector<Neighbor> &result,
                                    std::vector<float> &   occlude_factor) {
    if (pool.empty())
      return;
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(!pool.empty());

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree) {
      unsigned start = 0;

      while (result.size() < degree && (start) < pool.size() && start < maxc) {
        auto &p = pool[start];
        if (occlude_factor[start] > cur_alpha) {
          start++;
          continue;
        }
        occlude_factor[start] = std::numeric_limits<float>::max();
        result.push_back(p);
        for (unsigned t = start + 1; t < pool.size() && t < maxc; t++) {
          if (occlude_factor[t] > alpha)
            continue;
          float djk = _distance->compare(
              _data + _aligned_dim * (size_t) pool[t].id,
              _data + _aligned_dim * (size_t) p.id, (unsigned) _aligned_dim);
          occlude_factor[t] =
              (std::max)(occlude_factor[t], pool[t].distance / djk);
        }
        start++;
      }
      cur_alpha *= 1.2;
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::occlude_pq(std::vector<Neighbor> &pool,
                                  const unsigned location, const float alpha,
                                  const unsigned degree, const unsigned maxc,
                                  std::vector<Neighbor> &result) {
    auto               pool_size = (_u32) pool.size();
    std::vector<float> occlude_factor(pool_size, 0);
    occlude_list(pool, location, alpha, degree, maxc, result, occlude_factor);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::occlude_pq(std::vector<Neighbor> &pool,
                                  const unsigned location, const float alpha,
                                  const unsigned degree, const unsigned maxc,
                                  std::vector<Neighbor> &result,
                                  std::vector<float> &   occlude_factor) {
    if (pool.empty())
      return;
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(!pool.empty());

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree) {
      unsigned start = 0;

      while (result.size() < degree && (start) < pool.size() && start < maxc) {
        auto &p = pool[start];
        if (occlude_factor[start] > cur_alpha) {
          start++;
          continue;
        }
        occlude_factor[start] = std::numeric_limits<float>::max();
        result.push_back(p);
        for (unsigned t = start + 1; t < pool.size() && t < maxc; t++) {
          if (occlude_factor[t] > alpha)
            continue;
          unsigned p_id, q_id;
          p_id = pool[t].id;
          q_id = p.id;
          if (p_id == _max_points)
            p_id = _nd;
          if (q_id == _max_points)
            q_id = _nd;
          float djk = _distance->compare(
              _pq_data + _aligned_dim * (size_t) p_id,
              _pq_data + _aligned_dim * (size_t) q_id, (unsigned) _aligned_dim);
          occlude_factor[t] =
              (std::max)(occlude_factor[t], pool[t].distance / djk);
        }
        start++;
      }
      cur_alpha *= 1.2;
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::prune_neighbors(const unsigned location,
                                       std::vector<Neighbor> &pool,
                                       const Parameters &     parameter,
                                       std::vector<unsigned> &pruned_list) {
    unsigned range = parameter.Get<unsigned>("R");
    unsigned maxc = parameter.Get<unsigned>("C");
    float    alpha = parameter.Get<float>("alpha");

    if (pool.size() == 0)
      return;

    _width = (std::max)(_width, range);

    // sort the pool based on distance to query
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    result.reserve(range);
    std::vector<float> occlude_factor(pool.size(), 0);

    occlude_list(pool, location, alpha, range, maxc, result, occlude_factor);

    /* Add all the nodes in result into a variable called cut_graph
     * So this contains all the neighbors of id location
     */
    pruned_list.clear();
    assert(result.size() <= range);
    for (auto iter : result) {
      if (iter.id != location)
        pruned_list.emplace_back(iter.id);
    }

    if (_saturate_graph && alpha > 1) {
      for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
        if ((std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) ==
             pruned_list.end()) &&
            pool[i].id != location)
          pruned_list.emplace_back(pool[i].id);
      }
    }
  }

  /* batch_inter_insert():
   * This function tries to add reverse links from all the visited nodes to
   * the current node n.
   */
  template<typename T, typename TagT>
  void Index<T, TagT>::batch_inter_insert(
      unsigned n, const std::vector<unsigned> &pruned_list,
      const Parameters &parameter, std::vector<unsigned> &need_to_sync) {
    const auto range = parameter.Get<unsigned>("R");

    // assert(!src_pool.empty());

    for (auto des : pruned_list) {
      if (des == n)
        continue;
      /* des.id is the id of the neighbors of n */
      assert(des >= 0 && des < _max_points + _num_frozen_pts);
      if (des > _max_points)
        std::cout << "error. " << des << " exceeds max_pts" << std::endl;
      /* des_pool contains the neighbors of the neighbors of n */

      {
        LockGuard guard(_locks[des]);
        if (std::find(_final_graph[des].begin(), _final_graph[des].end(), n) ==
            _final_graph[des].end()) {
          _final_graph[des].push_back(n);
          if (_final_graph[des].size() > (unsigned) (range * SLACK_FACTOR))
            need_to_sync[des] = 1;
        }
      }  // des lock is released by this point
    }
  }

  /* inter_insert():
   * This function tries to add reverse links from all the visited nodes to
   * the current node n.
   */
  template<typename T, typename TagT>
  void Index<T, TagT>::inter_insert(unsigned n,
                                    std::vector<unsigned> &pruned_list,
                                    const Parameters &     parameter,
                                    bool                   update_in_graph) {
    const auto range = parameter.Get<unsigned>("R");
    assert(n >= 0 && n < _nd + _num_frozen_pts);
    const auto &src_pool = pruned_list;

    assert(!src_pool.empty());

    for (auto des : src_pool) {
      /* des.id is the id of the neighbors of n */
      assert(des >= 0 && des < _max_points + _num_frozen_pts);
      /* des_pool contains the neighbors of the neighbors of n */
      auto &                des_pool = _final_graph[des];
      std::vector<unsigned> copy_of_neighbors;
      bool                  prune_needed = false;
      {
        LockGuard guard(_locks[des]);
        if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
          if (des_pool.size() < SLACK_FACTOR * range) {
            des_pool.emplace_back(n);
            if (update_in_graph) {
              LockGuard guard(_locks_in[n]);
              _in_graph[n].emplace_back(des);
            }
            prune_needed = false;
          } else {
            copy_of_neighbors = des_pool;
            prune_needed = true;
          }
        }
      }  // des lock is released by this point

      if (prune_needed) {
        copy_of_neighbors.push_back(n);
        tsl::robin_set<unsigned> dummy_visited(0);
        std::vector<Neighbor>    dummy_pool(0);

        size_t reserveSize = (size_t)(std::ceil(1.05 * SLACK_FACTOR * range));
        dummy_visited.reserve(reserveSize);
        dummy_pool.reserve(reserveSize);

        for (auto cur_nbr : copy_of_neighbors) {
          if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
              cur_nbr != des) {
            float dist =
                _distance->compare(_data + _aligned_dim * (size_t) des,
                                   _data + _aligned_dim * (size_t) cur_nbr,
                                   (unsigned) _aligned_dim);
            dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
            dummy_visited.insert(cur_nbr);
          }
        }
        std::vector<unsigned> new_out_neighbors;
        prune_neighbors(des, dummy_pool, parameter, new_out_neighbors);
        {
          LockGuard guard(_locks[des]);
          // updating in_graph of out-neighbors of des
          if (update_in_graph) {
            for (auto out_nbr : _final_graph[des]) {
              {
                LockGuard guard(_locks_in[out_nbr]);
                for (unsigned i = 0; i < _in_graph[out_nbr].size(); i++) {
                  if (_in_graph[out_nbr][i] == des) {
                    _in_graph[out_nbr].erase(_in_graph[out_nbr].begin() + i);
                    break;
                  }
                }
              }
            }
          }

          _final_graph[des].clear();
          for (auto new_nbr : new_out_neighbors) {
            _final_graph[des].emplace_back(new_nbr);
            if (update_in_graph) {
              LockGuard guard(_locks_in[new_nbr]);
              _in_graph[new_nbr].emplace_back(des);
            }
          }
        }
      }
    }
  }
  /* Link():
   * The graph creation function.
   *    The graph will be updated periodically in NUM_SYNCS batches
   */
  template<typename T, typename TagT>
  void Index<T, TagT>::link(Parameters &parameters) {
    unsigned NUM_THREADS = parameters.Get<unsigned>("num_threads");
    if (NUM_THREADS != 0)
      omp_set_num_threads(NUM_THREADS);

    uint32_t NUM_SYNCS =
        (unsigned) DIV_ROUND_UP(_nd + _num_frozen_pts, (64 * 64));
    if (NUM_SYNCS < 40)
      NUM_SYNCS = 40;
    // std::cout << "Number of syncs: " << NUM_SYNCS << std::endl;

    _saturate_graph = parameters.Get<bool>("saturate_graph");

    if (NUM_THREADS != 0)
      omp_set_num_threads(NUM_THREADS);

    const unsigned argL = parameters.Get<unsigned>("L");  // Search list size
    const unsigned range = parameters.Get<unsigned>("R");
    const float    last_round_alpha = parameters.Get<float>("alpha");
    unsigned       L = argL;

    std::vector<unsigned> Lvec;
    Lvec.push_back(L);
    Lvec.push_back(L);
    const unsigned NUM_RNDS = 2;

    // Max degree of graph
    // Pruning parameter
    // Set alpha=1 for the first pass; use specified alpha for last pass
    parameters.Set<float>("alpha", 1);

    /* visit_order is a vector that is initialized to the entire graph */
    std::vector<unsigned>          visit_order;
    std::vector<diskann::Neighbor> pool, tmp;
    tsl::robin_set<unsigned>       visited;
    visit_order.reserve(_nd + _num_frozen_pts);
    for (unsigned i = 0; i < (unsigned) _nd; i++) {
      visit_order.emplace_back(i);
    }

    if (_num_frozen_pts > 0)
      visit_order.emplace_back((unsigned) _max_points);

    // if there are frozen points, the first such one is set to be the _ep
    if (_num_frozen_pts > 0)
      _ep = (unsigned) _max_points;
    else
      _ep = calculate_entry_point();

    /*    _final_graph.reserve(_max_points + _num_frozen_pts);
        _final_graph.resize(_max_points + _num_frozen_pts); */

    if (_support_eager_delete) {
      _in_graph.reserve(_max_points + _num_frozen_pts);
      _in_graph.resize(_max_points + _num_frozen_pts);
    }

    for (uint64_t p = 0; p < _nd; p++) {
      _final_graph[p].reserve((size_t)(std::ceil(range * SLACK_FACTOR * 1.05)));
    }

    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // creating a initial list to begin the search process. it has _ep and
    // random other nodes
    std::set<unsigned> unique_start_points;
    unique_start_points.insert(_ep);

    std::vector<unsigned> init_ids;
    for (auto pt : unique_start_points)
      init_ids.emplace_back(pt);

    diskann::Timer link_timer;
    for (uint32_t rnd_no = 0; rnd_no < NUM_RNDS; rnd_no++) {
      L = Lvec[rnd_no];

      if (rnd_no == NUM_RNDS - 1) {
        if (last_round_alpha > 1)
          parameters.Set<float>("alpha", last_round_alpha);
      }

      double   sync_time = 0, total_sync_time = 0;
      double   inter_time = 0, total_inter_time = 0;
      size_t   inter_count = 0, total_inter_count = 0;
      unsigned progress_counter = 0;

      size_t round_size = DIV_ROUND_UP(_nd, NUM_SYNCS);  // size of each batch
      std::vector<unsigned> need_to_sync(_max_points + _num_frozen_pts, 0);

      std::vector<std::vector<unsigned>> pruned_list_vector(round_size);

      for (uint32_t sync_num = 0; sync_num < NUM_SYNCS; sync_num++) {
        size_t start_id = sync_num * round_size;
        size_t end_id =
            (std::min)(_nd + _num_frozen_pts, (sync_num + 1) * round_size);

        auto s = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff;

#pragma omp parallel for schedule(dynamic)
        for (_s64 node_ctr = (_s64) start_id; node_ctr < (_s64) end_id;
             ++node_ctr) {
          auto                     node = visit_order[node_ctr];
          size_t                   node_offset = node_ctr - start_id;
          tsl::robin_set<unsigned> visited;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          // get nearest neighbors of n in tmp. pool contains all the
          // points that were checked along with their distance from
          // n. visited contains all the points visited, just the ids
          std::vector<Neighbor> pool;
          pool.reserve(L * 2);
          visited.reserve(L * 2);
          get_expanded_nodes(node, L, init_ids, pool, visited);
          /* check the neighbors of the query that are not part of
           * visited, check their distance to the query, and add it to
           * pool.
           */
          if (!_final_graph[node].empty())
            for (auto id : _final_graph[node]) {
              if (visited.find(id) == visited.end() && id != node) {
                float dist =
                    _distance->compare(_data + _aligned_dim * (size_t) node,
                                       _data + _aligned_dim * (size_t) id,
                                       (unsigned) _aligned_dim);
                pool.emplace_back(Neighbor(id, dist, true));
                visited.insert(id);
              }
            }
          prune_neighbors(node, pool, parameters, pruned_list);
        }
        diff = std::chrono::high_resolution_clock::now() - s;
        sync_time += diff.count();

// prune_neighbors will check pool, and remove some of the points and
// create a cut_graph, which contains neighbors for point n
#pragma omp parallel for schedule(dynamic, 64)
        for (_s64 node_ctr = (_s64) start_id; node_ctr < (_s64) end_id;
             ++node_ctr) {
          _u64                   node = visit_order[node_ctr];
          size_t                 node_offset = node_ctr - start_id;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          _final_graph[node].clear();
          for (auto id : pruned_list)
            _final_graph[node].emplace_back(id);
        }
        s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 64)
        for (_s64 node_ctr = start_id; node_ctr < (_s64) end_id; ++node_ctr) {
          auto                   node = visit_order[node_ctr];
          _u64                   node_offset = node_ctr - start_id;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          batch_inter_insert(node, pruned_list, parameters, need_to_sync);
          //          inter_insert(node, pruned_list, parameters, 0);
          pruned_list.clear();
          pruned_list.shrink_to_fit();
        }

#pragma omp parallel for schedule(dynamic, 65536)
        for (_s64 node_ctr = 0; node_ctr < (_s64)(visit_order.size());
             node_ctr++) {
          auto node = visit_order[node_ctr];
          if (need_to_sync[node] != 0) {
            need_to_sync[node] = 0;
            inter_count++;
            tsl::robin_set<unsigned> dummy_visited(0);
            std::vector<Neighbor>    dummy_pool(0);
            std::vector<unsigned>    new_out_neighbors;

            for (auto cur_nbr : _final_graph[node]) {
              if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
                  cur_nbr != node) {
                float dist =
                    _distance->compare(_data + _aligned_dim * (size_t) node,
                                       _data + _aligned_dim * (size_t) cur_nbr,
                                       (unsigned) _aligned_dim);
                dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
                dummy_visited.insert(cur_nbr);
              }
            }
            prune_neighbors(node, dummy_pool, parameters, new_out_neighbors);

            _final_graph[node].clear();
            for (auto id : new_out_neighbors)
              _final_graph[node].emplace_back(id);
          }
        }

        diff = std::chrono::high_resolution_clock::now() - s;
        inter_time += diff.count();

        if ((sync_num * 100) / NUM_SYNCS > progress_counter) {
          std::cout.precision(4);
          /*std::cout << "Completed  (round: " << rnd_no << ", sync: " <<
             sync_num
                    << "/" << NUM_SYNCS << " with L " << L << ")"
                    << " sync_time: " << sync_time << "s"
                    << "; inter_time: " << inter_time << "s" << std::endl;*/

          total_sync_time += sync_time;
          total_inter_time += inter_time;
          total_inter_count += inter_count;
          sync_time = 0;
          inter_time = 0;
          inter_count = 0;
          progress_counter += 5;
        }
      }

      MallocExtension::instance()->ReleaseFreeMemory();
      if (_nd > 0) {
        std::cout << "Completed Pass " << rnd_no << " of data using L=" << L
                  << " and alpha=" << parameters.Get<float>("alpha")
                  << ". Stats: ";
        std::cout << "search+prune_time=" << total_sync_time
                  << "s, inter_time=" << total_inter_time
                  << "s, inter_count=" << total_inter_count << std::endl;
      }
    }

    if (_nd > 0) {
      std::cout << "Starting final cleanup.." << std::flush;
    }
#pragma omp parallel for schedule(dynamic, 65536)
    for (_s64 node_ctr = 0; node_ctr < (_s64)(visit_order.size()); node_ctr++) {
      auto node = visit_order[node_ctr];
      if (_final_graph[node].size() > range) {
        tsl::robin_set<unsigned> dummy_visited(0);
        std::vector<Neighbor>    dummy_pool(0);
        std::vector<unsigned>    new_out_neighbors;

        for (auto cur_nbr : _final_graph[node]) {
          if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
              cur_nbr != node) {
            float dist =
                _distance->compare(_data + _aligned_dim * (size_t) node,
                                   _data + _aligned_dim * (size_t) cur_nbr,
                                   (unsigned) _aligned_dim);
            dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
            dummy_visited.insert(cur_nbr);
          }
        }
        prune_neighbors(node, dummy_pool, parameters, new_out_neighbors);

        _final_graph[node].clear();
        for (auto id : new_out_neighbors)
          _final_graph[node].emplace_back(id);
      }
    }
    if (_nd > 0) {
      std::cout << "done. Link time: "
                << ((double) link_timer.elapsed() / (double) 1000000) << "s"
                << std::endl;
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::prune_all_nbrs(const Parameters &parameters) {
    const unsigned range = parameters.Get<unsigned>("R");

    diskann::Timer timer;
#pragma omp        parallel for
    for (size_t node = 0; node < _max_points + _num_frozen_pts; node++) {
      if (node < _nd || node == _max_points) {
        if (_final_graph[node].size() > range) {
          tsl::robin_set<unsigned> dummy_visited(0);
          std::vector<Neighbor>    dummy_pool(0);
          std::vector<unsigned>    new_out_neighbors;

          for (auto cur_nbr : _final_graph[node]) {
            if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
                cur_nbr != node) {
              float dist =
                  _distance->compare(_data + _aligned_dim * (size_t) node,
                                     _data + _aligned_dim * (size_t) cur_nbr,
                                     (unsigned) _aligned_dim);
              dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
              dummy_visited.insert(cur_nbr);
            }
          }
          prune_neighbors(node, dummy_pool, parameters, new_out_neighbors);

          _final_graph[node].clear();
          for (auto id : new_out_neighbors)
            _final_graph[node].emplace_back(id);
        }
      }
    }

    std::cout << "Prune time : " << timer.elapsed() / 1000 << "ms" << std::endl;
    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < (_nd + _num_frozen_pts); i++) {
      auto &pool = _final_graph[i];
      max = (std::max)(max, pool.size());
      min = (std::min)(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    if (min > max)
      min = max;
    if (_nd > 0) {
      std::cout << "Index built with degree: max:" << max
                << "  avg:" << (float) total / (float) (_nd + _num_frozen_pts)
                << "  min:" << min << "  count(deg<2):" << cnt << std::endl;
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(const char *filename,
                             const size_t             num_points_to_load,
                             Parameters &             parameters,
                             const std::vector<TagT> &tags) {
    if (filename == nullptr) {
      std::cout << "Starting with an empty index." << std::endl;
      _nd = 0;
    } else

        if (!file_exists(filename)) {
      std::cerr << "Data file provided does not exist!!! Exiting...."
                << std::endl;
      exit(-1);
    }
    size_t file_dim, file_num_points;
    load_aligned_data<T>(std::string(filename), _data, file_num_points,
                         file_dim, _aligned_dim);

    if (file_dim != _dim) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << _dim << " dimension,"
             << "but file has " << file_dim << " dimension." << std::endl;
      std::cerr << stream.str() << std::endl;
      free(_data);
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
      free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    _nd = num_points_to_load;
    if (_enable_tags) {
      for (size_t i = 0; i < tags.size(); ++i) {
        _tag_to_location[tags[i]] = (unsigned) i;
        _location_to_tag[(unsigned) i] = tags[i];
      }
    }
    generate_frozen_point();
    link(parameters);  // Primary func for creating nsg graph

    if (_support_eager_delete) {
      update_in_graph();  // copying values to in_graph
    }

    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < _nd; i++) {
      auto &pool = _final_graph[i];
      max = (std::max)(max, pool.size());
      min = (std::min)(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    if (min > max)
      min = max;
    if (_nd > 0) {
      std::cout << "Index built with degree: max:" << max
                << "  avg:" << (float) total / (float) (_nd + _num_frozen_pts)
                << "  min:" << min << "  count(deg<2):" << cnt << std::endl;
    }
    _width = (std::max)((unsigned) max, _width);
    _has_built = true;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::check_graph_quality(_u32 num_samples,
                                           Parameters &parameters) {
    if (num_samples > _nd)
      num_samples = _nd;

    tsl::robin_set<_u32> samples;
    std::vector<_u32>    samples_vec;

    while (samples.size() < num_samples) {
      _u32 rnd_loc = rand() % _nd;
      if (samples.find(rnd_loc) == samples.end()) {
        samples.insert(rnd_loc);
        samples_vec.emplace_back(rnd_loc);
      }
    }
    _u32 good_count = 0;
    _u32 good_count1 = 0;
    _u32 good_count2 = 0;

#pragma omp parallel for schedule(dynamic, 64)
    for (_u32 i = 0; i < samples_vec.size(); i++) {
      auto                     location = samples_vec[i];
      std::vector<Neighbor>    pool;
      std::vector<Neighbor>    tmp;
      tsl::robin_set<unsigned> visited;

      pool.clear();
      tmp.clear();
      visited.clear();
      std::vector<unsigned> pruned_list;
      unsigned              Lindex = parameters.Get<unsigned>("L");

      std::vector<unsigned> init_ids;
      get_expanded_nodes(location, Lindex, init_ids, pool, visited);

      std::sort(pool.begin(), pool.end());
      for (unsigned i = 0; i < pool.size(); i++)
        if (pool[i].id == (unsigned) location) {
          pool.erase(pool.begin() + i);
          visited.erase((unsigned) location);
          break;
        }

      //    float dist_graph_nn = _distance->compare(_data +
      //    _final_graph[location][0] * _aligned_dim, _data + location *
      //    _aligned_dim, _aligned_dim);
      //    float dist_pool_nn = _distance->compare(_data + pool[0].id *
      //    _aligned_dim, _data + location * _aligned_dim, _aligned_dim);

      //#pragma omp critical
      //    {
      //    std::cout<<location<<": graph_nn "<< dist_graph_nn << ", pool_nn: "
      //    << dist_pool_nn << std::endl;

      //}

      if (pool[0].id == _final_graph[location][0]) {
#pragma omp atomic
        good_count++;
      }

      if (std::find(_final_graph[location].begin(),
                    _final_graph[location].end(),
                    pool[0].id) != _final_graph[location].end()) {
#pragma omp atomic
        good_count1++;
      }
      if (visited.find(_final_graph[location][0]) != visited.end()) {
#pragma omp atomic
        good_count2++;
      }
    }

    std::cout << "Nearest neighbor matches first edge in graph " << good_count
              << "/" << num_samples << " times. Good count 1: " << good_count1
              << ", Good count 2: " << good_count2 << std::endl;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(const T *source_data, const size_t num_points,
                             Parameters &             parameters,
                             const std::vector<TagT> &tags) {
    if (source_data == nullptr) {
      if (num_points > 0) {
        std::cout << "Data not supplied for static index build. Exiting"
                  << std::endl;
        exit(-1);
      }
    }

    if (num_points > _max_points) {
      std::cout << "Cannot build index on more than " << _max_points
                << " points. Exiting" << std::endl;
      exit(-2);
    }
    if (_enable_tags) {
      for (size_t i = 0; i < tags.size(); ++i) {
        _tag_to_location[tags[i]] = (unsigned) i;
        _location_to_tag[(unsigned) i] = tags[i];
      }
    }
    _nd = num_points;
    for (size_t i = 0; i < _nd; i++) {
      memcpy((void *) (_data + (size_t) _aligned_dim * i),
             (void *) (source_data + (size_t) _dim * i), sizeof(T) * _dim);
    }

    generate_frozen_point();
    link(parameters);  // Primary func for creating nsg graph

    if (_support_eager_delete) {
      update_in_graph();  // copying values to in_graph
    }

    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < _nd; i++) {
      auto &pool = _final_graph[i];
      max = (std::max)(max, pool.size());
      min = (std::min)(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    if (min > max)
      min = max;
    if (_nd > 0) {
      std::cout << "Index built with degree: max:" << max
                << "  avg:" << (float) total / (float) (_nd + _num_frozen_pts)
                << "  min:" << min << "  count(deg<2):" << cnt << std::endl;
    }
    _width = (std::max)((unsigned) max, _width);
    _has_built = true;
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search(
      const T *query, const size_t K, const unsigned L,
      std::vector<Neighbor_Tag<TagT>> &best_K_tags) {
    assert(best_K_tags.size() == 0);
    std::vector<unsigned>    init_ids;
    tsl::robin_set<unsigned> visited(10 * L);
    std::vector<Neighbor>    best, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    if (init_ids.size() == 0) {
      init_ids.emplace_back(_ep);
    }

    T *    aligned_query;
    size_t allocSize = _aligned_dim * sizeof(T);
    alloc_aligned(((void **) &aligned_query), allocSize, 8 * sizeof(T));
    memset(aligned_query, 0, _aligned_dim * sizeof(T));
    memcpy(aligned_query, query, _dim * sizeof(T));
    auto retval =
        iterate_to_fixed_point(aligned_query, L, init_ids, expanded_nodes_info,
                               expanded_nodes_ids, best);

    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    for (auto iter : best) {
      if (_location_to_tag.find(iter.id) != _location_to_tag.end())
        best_K_tags.emplace_back(
            Neighbor_Tag<TagT>(_location_to_tag[iter.id], iter.distance));
      if (best_K_tags.size() == K)
        break;
    }
    aligned_free(aligned_query);
    return retval;
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search(const T *query,
                                                       const size_t   K,
                                                       const unsigned L,
                                                       unsigned *     indices) {
    std::vector<unsigned>    init_ids;
    tsl::robin_set<unsigned> visited(10 * L);
    std::vector<Neighbor>    best_L_nodes, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    if (init_ids.size() == 0) {
      init_ids.emplace_back(_ep);
    }
    T *    aligned_query;
    size_t allocSize = _aligned_dim * sizeof(T);
    alloc_aligned(((void **) &aligned_query), allocSize, 8 * sizeof(T));
    memset(aligned_query, 0, _aligned_dim * sizeof(T));
    memcpy(aligned_query, query, _dim * sizeof(T));
    auto retval =
        iterate_to_fixed_point(aligned_query, L, init_ids, expanded_nodes_info,
                               expanded_nodes_ids, best_L_nodes);

    size_t pos = 0;
    for (auto it : best_L_nodes) {
      if (it.id < _max_points) {
        indices[pos] = it.id;
      }
      pos++;
      if (pos == K)
        break;
    }
    aligned_free(aligned_query);
    return retval;
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search(
      const T *query, const uint64_t K, const unsigned L,
      std::vector<unsigned> init_ids, uint64_t *indices, float *distances) {
    tsl::robin_set<unsigned> visited(10 * L);
    std::vector<Neighbor>    best_L_nodes, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    if (init_ids.size() == 0) {
      init_ids.emplace_back(_ep);
    }
    T *    aligned_query;
    size_t allocSize = _aligned_dim * sizeof(T);
    alloc_aligned(((void **) &aligned_query), allocSize, 8 * sizeof(T));
    memset(aligned_query, 0, _aligned_dim * sizeof(T));
    memcpy(aligned_query, query, _dim * sizeof(T));
    auto retval = iterate_to_fixed_point(aligned_query, (unsigned) L, init_ids,
                                         expanded_nodes_info,
                                         expanded_nodes_ids, best_L_nodes);

    size_t pos = 0;
    for (auto it : best_L_nodes) {
      indices[pos] = it.id;
      distances[pos] = it.distance;
      pos++;
      if (pos == K)
        break;
    }
    aligned_free(aligned_query);
    return retval;
  }

  template<typename T, typename TagT>

  std::pair<uint32_t, uint32_t> Index<T, TagT>::search_with_tags(
      const T *query, const size_t K, const unsigned L, TagT *tags,
      unsigned frozen_pts, unsigned *indices_buffer) {
    const bool alloc = indices_buffer == NULL;
    auto       indices = alloc ? new unsigned[K] : indices_buffer;
    auto       ret = search(query, K, L, indices);
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    size_t                                    pos = 0;
    for (int i = 0; i < (int) K; ++i)
      if (_location_to_tag.find(indices[i]) != _location_to_tag.end()) {
        tags[pos] = _location_to_tag[indices[i]];
        pos++;
        if (pos == K)
          break;
      }

    if (alloc)
      delete[] indices;
    return ret;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::get_num_points() {
    return _nd;
  }

  template<typename T, typename TagT>
  T *Index<T, TagT>::get_data() {
    if (_num_frozen_pts > 0) {
      T *    ret_data = nullptr;
      size_t allocSize = _nd * _aligned_dim * sizeof(T);
      alloc_aligned(((void **) &ret_data), allocSize, 8 * sizeof(T));
      memset(ret_data, 0, _nd * _aligned_dim * sizeof(T));
      memcpy(ret_data, _data, _nd * _aligned_dim * sizeof(T));
      return ret_data;
    }
    return _data;
  }
  template<typename T, typename TagT>
  size_t Index<T, TagT>::return_max_points() {
    return _max_points;
  }

  /*************************************************
   *      Support for Incremental Update
   *************************************************/

  // in case we add ''frozen'' auxiliary points to the dataset, these are not
  // visible to external world, we generate them here and update our dataset
  template<typename T, typename TagT>
  int Index<T, TagT>::generate_frozen_point() {
    if (_num_frozen_pts == 0)
      return 0;

    if (_nd == 0) {
      memset(_data + (_max_points) *_aligned_dim, 0, _aligned_dim * sizeof(T));
      return 1;
    }
    size_t res = rand() % _nd;
    memcpy(_data + _max_points * _aligned_dim, _data + res * _aligned_dim,
           _aligned_dim * sizeof(T));
    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::enable_delete() {
    assert(!_can_delete);
    assert(_enable_tags);

    if (_can_delete) {
      std::cerr << "Delete already enabled" << std::endl;
      return -1;
    }
    if (!_enable_tags) {
      std::cerr << "Tags must be instantiated for deletions" << std::endl;
      return -2;
    }

    if (_data_compacted) {
      assert(_empty_slots.size() == 0);

      for (unsigned slot = (unsigned) _nd; slot < _max_points; ++slot)
        _empty_slots.insert(slot);
      _data_compacted = false;
    }

    _lazy_done = false;
    _eager_done = false;
    _can_delete = true;

    if (_support_eager_delete) {
      _in_graph.resize(_max_points + _num_frozen_pts);
      _in_graph.reserve(_max_points + _num_frozen_pts);
      update_in_graph();
    }
    return 0;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::release_location() {
    LockGuard guard(_change_lock);
    _nd--;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::eager_delete(const TagT tag, const Parameters &parameters,
                                   int delete_mode) {
    if (_lazy_done && (!_data_compacted)) {
      std::cout << "Lazy delete requests issued but data not consolidated, "
                   "cannot proceed with eager deletes."
                << std::endl;
      return -1;
    }

    {
      std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
      if (_tag_to_location.find(tag) == _tag_to_location.end()) {
        std::cerr << "Delete tag not found" << std::endl;
        return -1;
      }
    }

    unsigned id = _tag_to_location[tag];

    {
      std::unique_lock<std::shared_timed_mutex> lock(_tag_lock);
      _location_to_tag.erase(_tag_to_location[tag]);
      _tag_to_location.erase(tag);
    }

    {
      std::unique_lock<std::shared_timed_mutex> lock(_delete_lock);
      _delete_set.insert(id);
      _empty_slots.insert(id);
    }

    const unsigned range = parameters.Get<unsigned>("R");
    const unsigned maxc = parameters.Get<unsigned>("C");
    const float    alpha = parameters.Get<float>("alpha");

    // delete point from out-neighbors' in-neighbor list
    {
      LockGuard guard(_locks[id]);
      for (size_t i = 0; i < _final_graph[id].size(); i++) {
        unsigned j = _final_graph[id][i];
        {
          LockGuard guard(_locks_in[j]);
          for (unsigned k = 0; k < _in_graph[j].size(); k++) {
            if (_in_graph[j][k] == id) {
              _in_graph[j].erase(_in_graph[j].begin() + k);
              break;
            }
          }
        }
      }
    }

    tsl::robin_set<unsigned> in_nbr;
    {
      LockGuard guard(_locks_in[id]);
      for (unsigned i = 0; i < _in_graph[id].size(); i++)
        in_nbr.insert(_in_graph[id][i]);
    }
    assert(_in_graph[id].size() == in_nbr.size());

    std::vector<Neighbor>    pool, tmp;
    tsl::robin_set<unsigned> visited;
    std::vector<unsigned>    intersection;
    unsigned                 Lindex = parameters.Get<unsigned>("L");
    std::vector<unsigned>    init_ids;
    if (delete_mode == 2) {
      // constructing list of in-neighbors to be processed
      get_expanded_nodes(id, Lindex, init_ids, pool, visited);

      /*      for (unsigned i = 0; i < pool.size(); i++)
              if (pool[i].id == id) {
                pool.erase(pool.begin() + i);
                break;
              }*/
      for (auto node : visited) {
        if (in_nbr.find(node) != in_nbr.end()) {
          intersection.push_back(node);
        }
      }
    }

    // deleting deleted point from all in-neighbors' out-neighbor list
    for (auto it : in_nbr) {
      LockGuard guard(_locks[it]);
      _final_graph[it].erase(
          std::remove(_final_graph[it].begin(), _final_graph[it].end(), id),
          _final_graph[it].end());
    }

    tsl::robin_set<unsigned> candidate_set;
    std::vector<Neighbor>    expanded_nghrs;
    std::vector<Neighbor>    result;

    for (size_t i = 0; i < intersection.size(); i++) {
      auto ngh = intersection[i];

      candidate_set.clear();
      expanded_nghrs.clear();
      result.clear();

      {
        std::shared_lock<std::shared_timed_mutex> lock(_delete_lock);
        if (_delete_set.find(ngh) != _delete_set.end())
          continue;
      }

      {
        LockGuard guard(_locks[ngh]);

        // constructing candidate set from out-neighbors and out-neighbors of
        // ngh and id
        {  // should a shared reader lock on delete_lock be held here at the
           // beginning of the two for loops or should it be held and release
           // for ech iteration of the for loops? Which is faster?

          std::shared_lock<std::shared_timed_mutex> lock(_delete_lock);
          for (auto j : _final_graph[id]) {
            if ((j != id) && (j != ngh) &&
                (_delete_set.find(j) == _delete_set.end()))
              candidate_set.insert(j);
          }

          for (auto j : _final_graph[ngh]) {
            if ((j != id) && (j != ngh) &&
                (_delete_set.find(j) == _delete_set.end()))
              candidate_set.insert(j);
          }
        }

        for (auto j : candidate_set)
          expanded_nghrs.push_back(
              Neighbor(j,
                       _distance->compare(_data + _aligned_dim * (size_t) ngh,
                                          _data + _aligned_dim * (size_t) j,
                                          (unsigned) _aligned_dim),
                       true));
        std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
        occlude_list(expanded_nghrs, ngh, alpha, range, maxc, result);

        // deleting ngh from its old out-neighbors' in-neighbor list
        for (auto iter : _final_graph[ngh]) {
          {
            LockGuard guard(_locks_in[iter]);
            for (unsigned k = 0; k < _in_graph[iter].size(); k++) {
              if (_in_graph[iter][k] == ngh) {
                _in_graph[iter].erase(_in_graph[iter].begin() + k);
                break;
              }
            }
          }
        }

        _final_graph[ngh].clear();

        // updating out-neighbors and in-neighbors of ngh
        {
          std::shared_lock<std::shared_timed_mutex> lock(_delete_lock);
          for (size_t i = 0; i < result.size(); i++) {
            auto j = result[i];
            if (_delete_set.find(j.id) == _delete_set.end()) {
              _final_graph[ngh].push_back(j.id);
              {
                LockGuard guard(_locks_in[j.id]);
                if (std::find(_in_graph[j.id].begin(), _in_graph[j.id].end(),
                              ngh) == _in_graph[j.id].end()) {
                  _in_graph[j.id].emplace_back(ngh);
                }
              }
            }
          }
        }
      }
    }

    _final_graph[id].clear();
    _in_graph[id].clear();

    release_location();

    _eager_done = true;
    return 0;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::update_in_graph() {
    //  std::cout << "Updating in_graph.....";
    for (unsigned i = 0; i < _in_graph.size(); i++)
      _in_graph[i].clear();

    for (size_t i = 0; i < _final_graph.size();
         i++)  // copying to in-neighbor graph
      for (size_t j = 0; j < _final_graph[i].size(); j++)
        _in_graph[_final_graph[i][j]].emplace_back((_u32) i);
  }

  // Do not call consolidate_deletes() if you have not locked _change_lock.
  // Returns number of live points left after consolidation
  // proxy inserts all nghrs of deleted points
  /*template<typename T, typename TagT>
  size_t Index<T, TagT>::consolidate_deletes(const Parameters &parameters) {
    if (_eager_done) {
      return 0;
    }

    assert(!_data_compacted);
    assert(_can_delete);
    assert(_enable_tags);
    assert(_delete_set.size() <= _nd);
    assert(_empty_slots.size() + _nd == _max_points);

    const unsigned range = parameters.Get<unsigned>("R");
    const unsigned maxc = parameters.Get<unsigned>("C");
    const float    alpha = parameters.Get<float>("alpha");

    std::vector<_u32>    nearby_undel_nbrs;
    std::vector<unsigned> del_vec;
    std::mutex            _lock_1;
    for (auto elem : _delete_set) {
      del_vec.push_back(elem);
    }

    for (size_t i = 0; i < del_vec.size();i++) {
      auto iter = del_vec[i];
      if (_empty_slots.find(iter) == _empty_slots.end()) {
        for (auto loc : _final_graph[iter]) {
          {
            LockGuard guard(_lock_1);
          if ((std::find(nearby_undel_nbrs.begin(), nearby_undel_nbrs.end(),
loc) == nearby_undel_nbrs.end()) && (_delete_set.find(loc) ==
_delete_set.end()))
          {
            nearby_undel_nbrs.emplace_back(loc);
          }
          }
        }
      }
    }

    _u64     total_pts = _max_points + _num_frozen_pts;
    unsigned block_size = 1 << 10;
    _s64     total_blocks = DIV_ROUND_UP(total_pts, block_size);

#pragma omp parallel for schedule(dynamic)
    for (_s64 block = 0; block < total_blocks; ++block) {
      tsl::robin_set<unsigned> candidate_set;
      std::vector<Neighbor>    expanded_nghrs;
      std::vector<Neighbor>    result;

      for (_s64 i = block * block_size;
           i < (_s64)((block + 1) * block_size) && i < (_s64)(total_pts); i++) {
        if (_delete_set.find(i) != _delete_set.end() ||
            _empty_slots.find(i) != _empty_slots.end())
          continue;

        candidate_set.clear();
        expanded_nghrs.clear();
        result.clear();

        bool modify = false;
            for (auto ngh : _final_graph[(_u32) i]) {
              if (_delete_set.find(ngh) != _delete_set.end()) {
                modify = true;

                // Add outgoing links from
                for (auto j : _final_graph[ngh])
                  if (_delete_set.find(j) == _delete_set.end())
                    candidate_set.insert(j);
              } else {
                candidate_set.insert(ngh);
              }
            }

            if (modify) {
              for (auto j : candidate_set)
                expanded_nghrs.push_back(
                    Neighbor(j, _distance->compare(_data + _aligned_dim * i,
_data + _aligned_dim * (size_t) j,
                    (unsigned) _aligned_dim), true));

              std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
              occlude_list(expanded_nghrs, (_u32) i, alpha, range, maxc,result);

              _final_graph[(_u32) i].clear();
              for (auto j : result) {
                if (j.id != (_u32) i)
                  _final_graph[(_u32) i].push_back(j.id);
              }
            }
           }
    }
        #pragma omp parallel for schedule(dynamic, 128)
    for (_u32 idx = 0; idx < nearby_undel_nbrs.size(); idx++) {
      unsigned                 Lindex = parameters.Get<unsigned>("L");
      std::vector<Neighbor>    pool;
      std::vector<Neighbor>    tmp;
      tsl::robin_set<unsigned> visited;
      std::vector<unsigned>    pruned_list;

      auto                  location = nearby_undel_nbrs[idx];
      std::vector<unsigned> init_ids;
      get_expanded_nodes(location, Lindex, init_ids, pool, visited);

      for (unsigned i = 0; i < pool.size(); i++)
        if (pool[i].id == (unsigned) location) {
          pool.erase(pool.begin() + i);
          visited.erase((unsigned) location);
          break;
        }

      prune_neighbors(location, pool, parameters, pruned_list);

      {
        LockGuard guard(_locks[location]);
        _final_graph[location].clear();
        _final_graph[location].shrink_to_fit();
        _final_graph[location].reserve((_u64)(range * SLACK_FACTOR * 1.05));
        for (auto link : pruned_list) {
          _final_graph[location].emplace_back(link);
        }
      }

      inter_insert(location, pruned_list, parameters, 0);
    }
    if (_support_eager_delete)
      update_in_graph();

    for (auto iter : _delete_set)
      _empty_slots.insert(iter);
    _nd -= _delete_set.size();
    return _nd;
  } */

  // takes only top 2r/4r of deleted points' out-nghrs as candidates in addition
  // to old nghrs
  /*  template<typename T, typename TagT>
    size_t Index<T, TagT>::consolidate_deletes(const Parameters &parameters) {
      if (_eager_done) {
        return 0;
      }

      assert(!_data_compacted);
      assert(_can_delete);
      assert(_enable_tags);
      assert(_delete_set.size() <= _nd);
      assert(_empty_slots.size() + _nd == _max_points);

      const unsigned range = parameters.Get<unsigned>("R");
      const unsigned maxc = parameters.Get<unsigned>("C");
      const float    alpha = parameters.Get<float>("alpha");

      _u64     total_pts = _max_points + _num_frozen_pts;
      unsigned block_size = 1 << 10;
      _s64     total_blocks = DIV_ROUND_UP(total_pts, block_size);

  #pragma omp parallel for schedule(dynamic)
      for (_s64 block = 0; block < total_blocks; ++block) {
        tsl::robin_set<unsigned> candidate_set_old;
        tsl::robin_set<unsigned> candidate_set_new;
        std::vector<Neighbor>    expanded_nghrs_old;
        std::vector<Neighbor>    expanded_nghrs_new;
        std::vector<Neighbor>    result;

        for (_s64 i = block * block_size;
             i < (_s64)((block + 1) * block_size) &&
             i < (_s64)(_max_points + _num_frozen_pts);
             i++) {
          if ((_delete_set.find((_u32) i) == _delete_set.end()) &&
              (_empty_slots.find((_u32) i) == _empty_slots.end())) {
            candidate_set_old.clear();
            candidate_set_new.clear();
            expanded_nghrs_old.clear();
            expanded_nghrs_new.clear();
            result.clear();

            bool modify = false;
            for (auto ngh : _final_graph[(_u32) i]) {
              if (_delete_set.find(ngh) != _delete_set.end()) {
                modify = true;

                // Add outgoing links from
                for (auto j : _final_graph[ngh])
                  if (_delete_set.find(j) == _delete_set.end())
                    candidate_set_new.insert(j);
              } else {
                candidate_set_old.insert(ngh);
              }
            }
            if (modify) {
              for (auto j : candidate_set_new)
                expanded_nghrs_new.push_back(
                    Neighbor(j,
                             _distance->compare(_data + _aligned_dim * i,
                                                _data + _aligned_dim * (size_t)
  j,
                                                (unsigned) _aligned_dim),
                             true));
              for (auto j : candidate_set_old)
                expanded_nghrs_old.push_back(
                    Neighbor(j,
                             _distance->compare(_data + _aligned_dim * i,
                                                _data + _aligned_dim * (size_t)
  j,
                                                (unsigned) _aligned_dim),
                             true));
              std::sort(expanded_nghrs_new.begin(), expanded_nghrs_new.end());
              size_t res = (2 * range);
              if (expanded_nghrs_new.size() < res) {
                expanded_nghrs_old.insert(expanded_nghrs_old.end(),
                                          expanded_nghrs_new.begin(),
                                          expanded_nghrs_new.end());
              } else {
                expanded_nghrs_old.insert(expanded_nghrs_old.end(),
                                          expanded_nghrs_new.begin(),
                                          expanded_nghrs_new.begin() + res);
              }
              std::sort(expanded_nghrs_old.begin(), expanded_nghrs_old.end());
              occlude_list(expanded_nghrs_old, (_u32) i, alpha, range, maxc,
                           result);

              _final_graph[(_u32) i].clear();
              for (auto j : result) {
                if (j.id != (_u32) i)
                  _final_graph[(_u32) i].push_back(j.id);
              }
            }
          }
        }
      }

      if (_support_eager_delete)
        update_in_graph();

      for (auto iter : _delete_set)
        _empty_slots.insert(iter);
      _nd -= _delete_set.size();
      return _nd;
    }
  */

  // proxy_insert approach
  /* template<typename T, typename TagT>
    size_t Index<T, TagT>::consolidate_deletes(const Parameters &parameters) {
      if (_eager_done) {
        return 0;
      }

      assert(!_data_compacted);
      assert(_can_delete);
      assert(_enable_tags);
      assert(_delete_set.size() <= _nd);
      assert(_empty_slots.size() + _nd == _max_points);

      const unsigned range = parameters.Get<unsigned>("R");
      const unsigned maxc = parameters.Get<unsigned>("C");
      const float    alpha = parameters.Get<float>("alpha");

      _u64     total_pts = _max_points + _num_frozen_pts;
      unsigned block_size = 1 << 10;
      _s64     total_blocks = DIV_ROUND_UP(total_pts, block_size);
      std::vector<unsigned> nearest_nghr, del_vec;
      std::mutex            _lock_1;
    for (auto elem : _delete_set) {
      del_vec.push_back(elem);
    }
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < del_vec.size(); i++) {
      unsigned elem = del_vec[i];*/
  /* std::vector<Neighbor>    pool;
   tsl::robin_set<unsigned> visited;
   std::vector<unsigned>    pruned_list;
   unsigned                 Lindex = parameters.Get<unsigned>("L");
   std::vector<unsigned> init_ids;

   get_expanded_nodes(elem, Lindex, init_ids, pool, visited);
   for (unsigned i = 0; i < pool.size(); i++)
   {
     if (pool[i].id == (unsigned) elem) {
           pool.erase(pool.begin() + i);
           visited.erase((unsigned) elem);
           break;
         }
   }
   unsigned min_idx;
   float    min_dist = std::numeric_limits<float>::max();
   for (size_t i = 0; i < pool.size(); i++) {
     if (pool[i].distance < min_dist)
     {
       min_dist = pool[i].distance;
       min_idx = i;
     }
   }
   unsigned entry = pool[min_idx].id; */
  /*unsigned entry = _final_graph[elem][0];
  {
    LockGuard guard(_lock_1);
    if (std::find(nearest_nghr.begin(), nearest_nghr.end(), entry) ==
        nearest_nghr.end()) {
      if (_delete_set.find(entry) == _delete_set.end()) {
        nearest_nghr.push_back(entry);
      }
    }
  }
}
#pragma omp parallel for schedule(dynamic)
  for (_s64 block = 0; block < total_blocks; ++block) {
    tsl::robin_set<unsigned> candidate_set;
    std::vector<Neighbor>    expanded_nghrs;
    std::vector<Neighbor>    result;

    for (_s64 i = block * block_size;
         i < (_s64)((block + 1) * block_size) &&
         i < (_s64)(_max_points + _num_frozen_pts);
         i++) {
      if ((_delete_set.find((_u32) i) == _delete_set.end()) &&
          (_empty_slots.find((_u32) i) == _empty_slots.end())) {
        candidate_set.clear();
        expanded_nghrs.clear();
        result.clear();

        bool modify = false;
        for (auto ngh : _final_graph[(_u32) i]) {
          if (_delete_set.find(ngh) != _delete_set.end()) {
            modify = true;

            // Add outgoing links from
            for (auto j : _final_graph[ngh])
              if (_delete_set.find(j) == _delete_set.end())
                candidate_set.insert(j);
          } else {
            candidate_set.insert(ngh);
          }
        }

        if (modify) {
          for (auto j : candidate_set)
            expanded_nghrs.push_back(
                Neighbor(j, _distance->compare(_data + _aligned_dim * i, _data +
_aligned_dim * (size_t) j,
                (unsigned) _aligned_dim), true));

          std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
          occlude_list(expanded_nghrs, (_u32) i, alpha, range, maxc,result);

          _final_graph[(_u32) i].clear();
          for (auto j : result) {
            if (j.id != (_u32) i)
              _final_graph[(_u32) i].push_back(j.id);
          }
        }
      }
    }
  }
  #pragma omp parallel for schedule(dynamic, 128)
for (_u32 idx = 0; idx < nearest_nghr.size(); idx++) {
  unsigned                 Lindex = parameters.Get<unsigned>("L");
  std::vector<Neighbor>    pool;
  std::vector<Neighbor>    tmp;
  tsl::robin_set<unsigned> visited;
  std::vector<unsigned>    pruned_list;

  auto                  location = nearest_nghr[idx];
  std::vector<unsigned> init_ids;
  get_expanded_nodes(location, Lindex, init_ids, pool, visited);

  for (unsigned i = 0; i < pool.size(); i++)
    if (pool[i].id == (unsigned) location) {
      pool.erase(pool.begin() + i);
      visited.erase((unsigned) location);
      break;
    }

  prune_neighbors(location, pool, parameters, pruned_list);

  {
    LockGuard guard(_locks[location]);
    _final_graph[location].clear();
    _final_graph[location].shrink_to_fit();
    _final_graph[location].reserve((_u64)(range * SLACK_FACTOR * 1.05));
    for (auto link : pruned_list) {
      _final_graph[location].emplace_back(link);
    }
  }

  inter_insert(location, pruned_list, parameters, 0);
}
  if (_support_eager_delete)
    update_in_graph();

  for (auto iter : _delete_set)
    _empty_slots.insert(iter);
  _nd -= _delete_set.size();
  return _nd;
}*/

  // using pq for distance calculations
  /*  template<typename T, typename TagT>
    size_t Index<T, TagT>::consolidate_deletes(const Parameters &parameters) {
      if (_eager_done) {
        return 0;
      }

      assert(!_data_compacted);
      assert(_can_delete);
      assert(_enable_tags);
      assert(_delete_set.size() <= _nd);
      assert(_empty_slots.size() + _nd == _max_points);

      const unsigned range = parameters.Get<unsigned>("R");
      const unsigned maxc = parameters.Get<unsigned>("C");
      const float    alpha = parameters.Get<float>("alpha");

      std::cout << "Loading pq vectors....";
      size_t      pq_npts, pq_dim;
      std::string bin_file =
          "/mnt/aditi/vectors_01_5M/float_data/vec_2.5M_0_lti_pq_compressed.bin_full.bin";
      diskann::load_bin<T>(bin_file, _pq_data, pq_npts, pq_dim);
      std::cout << "done" << std::endl;
      std::cout << pq_npts << " " << pq_dim << std::endl;
      _u64     total_pts = _max_points + _num_frozen_pts;
      unsigned block_size = 1 << 10;
      _s64     total_blocks = DIV_ROUND_UP(total_pts, block_size);

  #pragma omp parallel for schedule(dynamic)
      for (_s64 block = 0; block < total_blocks; ++block) {
        tsl::robin_set<unsigned> candidate_set;
        std::vector<Neighbor>    expanded_nghrs;
        std::vector<Neighbor>    result;

        for (_s64 i = block * block_size;
             i < (_s64)((block + 1) * block_size) &&
             i < (_s64)(_max_points + _num_frozen_pts);
             i++) {
          if ((_delete_set.find((_u32) i) == _delete_set.end()) &&
              (_empty_slots.find((_u32) i) == _empty_slots.end())) {
            candidate_set.clear();
            expanded_nghrs.clear();
            result.clear();

            bool modify = false;
            for (auto ngh : _final_graph[(_u32) i]) {
              if (_delete_set.find(ngh) != _delete_set.end()) {
                modify = true;

                // Add outgoing links from
                for (auto j : _final_graph[ngh])
                  if (_delete_set.find(j) == _delete_set.end())
                    candidate_set.insert(j);
              } else {
                candidate_set.insert(ngh);
              }
            }

            if (modify) {
              for (auto j : candidate_set) {
                unsigned j_temp = j;
                if (j >= _nd) {
                  if (j == _max_points) {
                    j_temp = _nd;
                  } else {
                    std::cout << "Invalid point found in graph  : " << j
                              << std::endl;
                    exit(-1);
                  }
                }

                unsigned i_temp = i;
                if (i >= _nd) {
                  if (i == _max_points)
                    i_temp = _nd;
                  else
                    exit(-1);
                }
                //              std::cout << j << " " << j_temp << std::endl;
                expanded_nghrs.push_back(Neighbor(
                    j,
                    _distance->compare(_pq_data + _aligned_dim * i_temp,
                                       _pq_data + _aligned_dim * (size_t)
  j_temp,
                                       (unsigned) _aligned_dim),
                    true));
              }

              std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
              occlude_pq(expanded_nghrs, (_u32) i, alpha, range, maxc, result);

              _final_graph[(_u32) i].clear();
              for (auto j : result) {
                if (j.id != (_u32) i)
                  _final_graph[(_u32) i].push_back(j.id);
              }
            }
          }
        }
      }
      if (_support_eager_delete)
        update_in_graph();

      for (auto iter : _delete_set)
        _empty_slots.insert(iter);
      _nd -= _delete_set.size();
      return _nd;
    }*/

  // original approach
  template<typename T, typename TagT>
  size_t Index<T, TagT>::consolidate_deletes(const Parameters &parameters) {
    if (_eager_done) {
      return 0;
    }

    assert(!_data_compacted);
    assert(_can_delete);
    assert(_enable_tags);
    assert(_delete_set.size() <= _nd);
    assert(_empty_slots.size() + _nd == _max_points);

    const unsigned range = parameters.Get<unsigned>("R");
    const unsigned maxc = parameters.Get<unsigned>("C");
    const float    alpha = parameters.Get<float>("alpha");

    _u64     total_pts = _max_points + _num_frozen_pts;
    unsigned block_size = 1 << 10;
    _s64     total_blocks = DIV_ROUND_UP(total_pts, block_size);

#pragma omp parallel for schedule(dynamic)
    for (_s64 block = 0; block < total_blocks; ++block) {
      tsl::robin_set<unsigned> candidate_set;
      std::vector<Neighbor>    expanded_nghrs;
      std::vector<Neighbor>    result;

      for (_s64 i = block * block_size;
           i < (_s64)((block + 1) * block_size) &&
           i < (_s64)(_max_points + _num_frozen_pts);
           i++) {
        if ((_delete_set.find((_u32) i) == _delete_set.end()) &&
            (_empty_slots.find((_u32) i) == _empty_slots.end())) {
          candidate_set.clear();
          expanded_nghrs.clear();
          result.clear();

          bool modify = false;
          for (auto ngh : _final_graph[(_u32) i]) {
            if (_delete_set.find(ngh) != _delete_set.end()) {
              modify = true;

              // Add outgoing links from
              for (auto j : _final_graph[ngh])
                if (_delete_set.find(j) == _delete_set.end())
                  candidate_set.insert(j);
            } else {
              candidate_set.insert(ngh);
            }
          }

          if (modify) {
            for (auto j : candidate_set) {
              expanded_nghrs.push_back(
                  Neighbor(j,
                           _distance->compare(_data + _aligned_dim * i,
                                              _data + _aligned_dim * (size_t) j,
                                              (unsigned) _aligned_dim),
                           true));
            }

            std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
            occlude_list(expanded_nghrs, (_u32) i, alpha, range, maxc, result);

            _final_graph[(_u32) i].clear();
            for (auto j : result) {
              if (j.id != (_u32) i)
                _final_graph[(_u32) i].push_back(j.id);
            }
          }
        }
      }
    }
    if (_support_eager_delete)
      update_in_graph();

    for (auto iter : _delete_set)
      _empty_slots.insert(iter);
    _nd -= _delete_set.size();
    return _nd;
  }

  // retain all old nghrs - send only nghrs of del points for occlude - to fill
  // in empty slots
  /* template<typename T, typename TagT>
   size_t Index<T, TagT>::consolidate_deletes(const Parameters &parameters) {
     if (_eager_done) {
       return 0;
     }

     assert(!_data_compacted);
     assert(_can_delete);
     assert(_enable_tags);
     assert(_delete_set.size() <= _nd);
     assert(_empty_slots.size() + _nd == _max_points);

     const unsigned range = parameters.Get<unsigned>("R");
     const unsigned maxc = parameters.Get<unsigned>("C");
     const float    alpha = parameters.Get<float>("alpha");

     _u64     total_pts = _max_points + _num_frozen_pts;
     unsigned block_size = 1 << 10;
     _s64     total_blocks = DIV_ROUND_UP(total_pts, block_size);

 #pragma omp parallel for schedule(dynamic)
     for (_s64 block = 0; block < total_blocks; ++block) {
       tsl::robin_set<unsigned> candidate_set_old;
       tsl::robin_set<unsigned> candidate_set_new;
       std::vector<Neighbor>    expanded_nghrs;
       std::vector<Neighbor>    result;

       for (_s64 i = block * block_size;
            i < (_s64)((block + 1) * block_size) &&
            i < (_s64)(_max_points + _num_frozen_pts);
            i++) {
         if ((_delete_set.find((_u32) i) == _delete_set.end()) &&
             (_empty_slots.find((_u32) i) == _empty_slots.end())) {
           candidate_set_new.clear();
           candidate_set_old.clear();
           expanded_nghrs.clear();
           result.clear();

           bool modify = false;
           for (auto ngh : _final_graph[(_u32) i]) {
             if (_delete_set.find(ngh) != _delete_set.end()) {
               modify = true;
               // Add outgoing links from
               for (auto j : _final_graph[ngh])
                 if (_delete_set.find(j) == _delete_set.end())
                   candidate_set_new.insert(j);
             } else {
               candidate_set_old.insert(ngh);
             }
           }

           if (modify) {
             for (auto elem : candidate_set_old) {
               if (candidate_set_new.find(elem) != candidate_set_new.end()) {
                 candidate_set_new.erase(elem);
               }
             }
             if (!candidate_set_new.empty()) {
               for (auto j : candidate_set_new)
                 expanded_nghrs.push_back(Neighbor(
                     j,
                     _distance->compare(_data + _aligned_dim * i,
                                        _data + _aligned_dim * (size_t) j,
                                        (unsigned) _aligned_dim),
                     true));

               std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
               occlude_list(expanded_nghrs, (_u32) i, alpha,
                            range - candidate_set_old.size(), maxc, result);
             }

             _final_graph[(_u32) i].clear();
             for (auto j : candidate_set_old) {
               if (j != i) {
                 _final_graph[i].push_back(j);
               }
               if (_final_graph[(_u32) i].size() == range) {
                 break;
               }
             }
             for (auto j : result) {
               if (j.id != (_u32) i) {
                 _final_graph[(_u32) i].push_back(j.id);
               }
               if (_final_graph[(_u32) i].size() == range) {
                 break;
               }
             }
           }
         }
       }
     }
     if (_support_eager_delete)
       update_in_graph();

     for (auto iter : _delete_set)
       _empty_slots.insert(iter);
     _nd -= _delete_set.size();
     return _nd;
   } */

  template<typename T, typename TagT>
  void Index<T, TagT>::compact_frozen_point() {
    if (_nd < _max_points) {
      if (_num_frozen_pts > 0) {
        // set new _ep to be frozen point
        _ep = _nd;
        if (!_final_graph[_max_points].empty()) {
          for (unsigned i = 0; i < _nd; i++)
            for (unsigned j = 0; j < _final_graph[i].size(); j++)
              if (_final_graph[i][j] == _max_points)
                _final_graph[i][j] = (_u32) _nd;

          _final_graph[_nd].clear();
          for (unsigned k = 0; k < _final_graph[_max_points].size(); k++)
            _final_graph[_nd].emplace_back(_final_graph[_max_points][k]);

          _final_graph[_max_points].clear();
          if (_support_eager_delete)
            update_in_graph();

          memcpy((void *) (_data + (size_t) _aligned_dim * _nd),
                 _data + (size_t) _aligned_dim * _max_points, sizeof(T) * _dim);
          memset((_data + (size_t) _aligned_dim * _max_points), 0,
                 sizeof(T) * _aligned_dim);
        }
      }
    }
    _ep = _nd;
  }

  // lazy deletion - processing selected in-neighbors obtained through search.
  /*  template<typename T, typename TagT>/
    size_t Index<T, TagT>::consolidate_eletes(const Parameters &parameters) {
      if (_eager_done) {
        return 0;
      }

      assert(!_compacted_lazy_deletions);
      assert(_can_delete);
      assert(_enable_tags);
      assert(_delete_set.size() <= _nd);
      assert(_empty_slots.size() + _nd == _max_points);

      const unsigned        range = parameters.Get<unsigned>("R");
      const unsigned        maxc = parameters.Get<unsigned>("C");
      const float           alpha = parameters.Get<float>("alpha");
      unsigned              Lindex = 50;
      std::vector<unsigned> init_ids;
      std::vector<unsigned> del_vec;
      std::unordered_map<unsigned, tsl::robin_set<unsigned>> visited_all;
      std::mutex _visited_lock;

      for (auto iter : _delete_set) {
        del_vec.push_back(iter);
      }
  #pragma omp parallel for schedule(dynamic)
      for (size_t i = 0; i < del_vec.size(); i++) {
        std::vector<Neighbor>    pool;
        tsl::robin_set<unsigned> visited;
        get_expanded_nodes(del_vec[i], Lindex, init_ids, pool, visited);
        {
          LockGuard guard(_visited_lock);
          visited_all[del_vec[i]] = visited;
        }
      }

      unsigned total_pts = _max_points + _num_frozen_pts;
      unsigned block_size = 1 << 10;
      unsigned total_blocks = DIV_ROUND_UP(total_pts, block_size);
      std::cout << "Start graph update operations." << std::endl;
  #pragma omp parallel for schedule(dynamic)
      for (unsigned block = 0; block < total_blocks; ++block) {
        tsl::robin_set<unsigned> candidate_set;
        std::vector<Neighbor>    expanded_nghrs;
        std::vector<Neighbor>    result;

        for (unsigned i = block * block_size;
             i < (block + 1) * block_size && i < _max_points + _num_frozen_pts;
             i++) {
          if ((_delete_set.find(i) == _delete_set.end()) &&
              (_empty_slots.find(i) == _empty_slots.end())) {
            candidate_set.clear();
            expanded_nghrs.clear();
            std::vector<unsigned> to_be_deleted;
            result.clear();

            bool modify = false;
            for (auto ngh : _final_graph[i]) {
              if (_delete_set.find(ngh) != _delete_set.end()) {
                if (visited_all[ngh].find(i) != visited_all[ngh].end()) {
                  modify = true;
                  for (auto j : _final_graph[ngh])
                    if (_delete_set.find(j) == _delete_set.end())
                      candidate_set.insert(j);
                } else {
                  to_be_deleted.push_back(ngh);
                }
              } else {
                candidate_set.insert(ngh);
              }
            }

            if (modify) {
              for (auto j : candidate_set)
                expanded_nghrs.push_back(
                    Neighbor(j,
                             _distance->compare(_data + _aligned_dim * (size_t)
  i,
                                                _data + _aligned_dim * (size_t)
  j,
                                                (unsigned) _aligned_dim),
                             true));
              std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
              occlude_list(expanded_nghrs, i, alpha, range, maxc, result);

              _final_graph[i].clear();
              for (auto j : result) {
                if (j.id != i)
                  _final_graph[i].push_back(j.id);
              }
            } else {
              for (auto iter : to_be_deleted) {
                _final_graph[i].erase(std::remove(_final_graph[i].begin(),
                                                  _final_graph[i].end(), iter),
                                      _final_graph[i].end());
              }
            }
          }
        }
      }

      if (_support_eager_delete)
        update_in_graph();

      for (auto iter : _delete_set)
        _empty_slots.insert(iter);
      _nd -= _delete_set.size();
      return _nd;
    }*/

  template<typename T, typename TagT>
  void Index<T, TagT>::get_active_tags(tsl::robin_set<unsigned> &active_tags) {
    for (auto iter : _tag_to_location) {
      active_tags.insert(iter.first);
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::compact_data_for_search() {
    compact_data();
    compact_frozen_point();
  }
  template<typename T, typename TagT>
  void Index<T, TagT>::compact_data_for_insert() {
    /*   if (_lazy_done && !_data_compacted)
         compact_data();
       else if (_eager_done && !_data_compacted)
         compact_data();*/
    compact_data();

    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < (_nd + _num_frozen_pts); i++) {
      auto &pool = _final_graph[i];
      max = (std::max)(max, pool.size());
      min = (std::min)(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    if (min > max)
      min = max;
    if (_nd > 0) {
      std::cout << "Index built with degree: max:" << max
                << "  avg:" << (float) total / (float) (_nd + _num_frozen_pts)
                << "  min:" << min << "  count(deg<2):" << cnt << std::endl;
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::compact_data() {
    //    std::cout << "Before compacting data, frozen point has  "
    //              << _final_graph[_max_points].size() << "nbrs " << std::endl;
    std::vector<unsigned> new_location = std::vector<unsigned>(
        _max_points + _num_frozen_pts, (_u32) _max_points);

    _u32 new_counter = 0;

    for (_u32 old_counter = 0; old_counter < _max_points + _num_frozen_pts;
         old_counter++) {
      if (_location_to_tag.find(old_counter) != _location_to_tag.end()) {
        new_location[old_counter] = new_counter;
        new_counter++;
      }
    }

    // If start node is removed, replace it.
    if (_delete_set.find(_ep) != _delete_set.end()) {
      std::cerr << "Replacing start node which has been deleted... "
                << std::flush;
      auto old_ep = _ep;
      // First active neighbor of old start node is new start node
      for (auto iter : _final_graph[_ep])
        if (_delete_set.find(iter) != _delete_set.end()) {
          _ep = iter;
          break;
        }
      if (_ep == old_ep) {
        std::cerr << "ERROR: Did not find a replacement for start node."
                  << std::endl;
        throw diskann::ANNException(
            "ERROR: Did not find a replacement for start node.", -1,
            __FUNCSIG__, __FILE__, __LINE__);
      } else {
        assert(_delete_set.find(_ep) == _delete_set.end());
        //        std::cout << "New start node is " << _ep << std::endl;
      }
    }

    //    std::cout << "Re-numbering nodes and edges and consolidating data... "
    //              << std::endl;
    for (unsigned old = 0; old <= _max_points; ++old) {
      if ((new_location[old] < _max_points) ||
          (old == _max_points)) {  // If point continues to exist

        // Renumber nodes to compact the order
        for (size_t i = 0; i < _final_graph[old].size(); ++i) {
          if (new_location[_final_graph[old][i]] > _final_graph[old][i]) {
            std::cout << "Error in compact_data(), old_location = "
                      << _final_graph[old][i] << ", new_location = "
                      << new_location[_final_graph[old][i]] << ", old = " << old
                      << ", new_location[old] = " << new_location[old]
                      << std::endl;
            std::cout << (_delete_set.find(
                              _location_to_tag[_final_graph[old][i]]) ==
                          _delete_set.end())
                      << std::endl;
            exit(-1);
          }
          _final_graph[old][i] = new_location[_final_graph[old][i]];
        }

        if (_support_eager_delete)
          for (size_t i = 0; i < _in_graph[old].size(); ++i) {
            if (new_location[_in_graph[old][i]] <= _in_graph[old][i])
              _in_graph[old][i] = new_location[_in_graph[old][i]];
          }

        // Move the data and adj list to the correct position
        if (new_location[old] != old) {
          assert(new_location[old] < old);
          _final_graph[new_location[old]].swap(_final_graph[old]);
          if (_support_eager_delete)
            _in_graph[new_location[old]].swap(_in_graph[old]);
          memcpy((void *) (_data + _aligned_dim * (size_t) new_location[old]),
                 (void *) (_data + _aligned_dim * (size_t) old),
                 _aligned_dim * sizeof(T));
        }
      } else
        _final_graph[old].clear();
    }
    //    std::cout << "done." << std::endl;

    //    std::cout << "Updating mapping between tags and ids... " << std::endl;
    // Update the location pointed to by tag
    _tag_to_location.clear();
    for (auto iter : _location_to_tag)
      _tag_to_location[iter.second] = new_location[iter.first];
    _location_to_tag.clear();
    for (auto iter : _tag_to_location)
      _location_to_tag[iter.second] = iter.first;

    //    std::cout << "After compacting data, frozen point has  "
    //              << _final_graph[_max_points].size() << "nbrs " << std::endl;

    for (_u64 old = _nd; old < _max_points; ++old)
      _final_graph[old].clear();
    _delete_set.clear();
    _empty_slots.clear();

    _lazy_done = false;
    _eager_done = false;
    _data_compacted = true;
    //    std::cout << "Consolidated the index" << std::endl;
  }

  // Do not call reserve_location() if you have not locked _change_lock.
  // It is not thread safe.
  template<typename T, typename TagT>
  int Index<T, TagT>::reserve_location() {
    LockGuard guard(_change_lock);
    if (_nd >= _max_points) {
      std::cerr << "Reached maximum capacity, cannot add more points"
                << std::endl;
      return -1;
    }
    unsigned location;
    if (_data_compacted)
      location = (unsigned) _nd;
    else {
      // no need of delete_lock here, _change_lock will ensure no other thread
      // executes this block of code
      assert(_empty_slots.size() != 0);
      assert(_empty_slots.size() + _nd == _max_points);

      auto iter = _empty_slots.begin();
      location = *iter;
      _empty_slots.erase(iter);
      _delete_set.erase(location);
    }

    ++_nd;

    return location;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::reposition_frozen_point_to_end() {
    if (_num_frozen_pts > 0) {
      if (_final_graph[_max_points].empty()) {
        for (unsigned i = 0; i < _nd; i++)
          for (unsigned j = 0; j < _final_graph[i].size(); j++)
            if (_final_graph[i][j] == _nd)
              _final_graph[i][j] = (unsigned) _max_points;

        _final_graph[_max_points].clear();
        for (unsigned k = 0; k < _final_graph[_nd].size(); k++)
          _final_graph[_max_points].emplace_back(_final_graph[_nd][k]);

        _final_graph[_nd].clear();

        if (_support_eager_delete)
          update_in_graph();
        memcpy((void *) (_data + (size_t) _aligned_dim * _max_points),
               _data + (size_t) _aligned_dim * _nd, sizeof(float) * _dim);
        memset((_data + (size_t) _aligned_dim * _nd), 0,
               sizeof(float) * _aligned_dim);
      }
    }
    _ep = (_u32) _max_points;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::insert_point(const T *point, const Parameters &parameters,
                                   const TagT tag) {
    unsigned range = parameters.Get<unsigned>("R");
    //    assert(_has_built);
    std::vector<Neighbor>    pool;
    std::vector<Neighbor>    tmp;
    tsl::robin_set<unsigned> visited;

    {
      std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
      if (_enable_tags &&
          (_tag_to_location.find(tag) != _tag_to_location.end())) {
        std::cerr << "Entry with the tag " << tag << " exists already"
                  << std::endl;
        exit(-1);
        return -1;
      }
    }

    auto location = reserve_location();
    if (location == -1) {
      std::cerr << "Can not insert, reached maximum(" << _max_points
                << ") points." << std::endl;
      return -2;
    }

    {
      std::unique_lock<std::shared_timed_mutex> lock(_tag_lock);
      _tag_to_location[tag] = location;
      _location_to_tag[location] = tag;
    }

    auto offset_data = _data + (size_t) _aligned_dim * location;
    memset((void *) offset_data, 0, sizeof(T) * _aligned_dim);
    memcpy((void *) offset_data, point, sizeof(T) * _dim);

    pool.clear();
    tmp.clear();
    visited.clear();
    std::vector<unsigned> pruned_list;
    unsigned              Lindex = parameters.Get<unsigned>("L");

    std::vector<unsigned> init_ids;
    get_expanded_nodes(location, Lindex, init_ids, pool, visited);

    for (unsigned i = 0; i < pool.size(); i++)
      if (pool[i].id == (unsigned) location) {
        pool.erase(pool.begin() + i);
        visited.erase((unsigned) location);
        break;
      }

    prune_neighbors(location, pool, parameters, pruned_list);
    assert(_final_graph.size() == _max_points + _num_frozen_pts);

    if (_support_eager_delete) {
      for (unsigned i = 0; i < _final_graph[location].size(); i++) {
        {
          LockGuard guard(_locks_in[_final_graph[location][i]]);
          _in_graph[_final_graph[location][i]].erase(
              std::remove(_in_graph[_final_graph[location][i]].begin(),
                          _in_graph[_final_graph[location][i]].end(), location),
              _in_graph[_final_graph[location][i]].end());
        }
      }
    }

    _final_graph[location].clear();
    _final_graph[location].shrink_to_fit();
    _final_graph[location].reserve((_u64)(range * SLACK_FACTOR * 1.05));
    assert(!pruned_list.empty());
    {
      LockGuard guard(_locks[location]);
      for (auto link : pruned_list) {
        _final_graph[location].emplace_back(link);
        if (_support_eager_delete) {
          LockGuard guard(_locks_in[link]);
          _in_graph[link].emplace_back(location);
        }
      }
    }

    assert(_final_graph[location].size() <= range);
    if (_support_eager_delete)
      inter_insert(location, pruned_list, parameters, 1);
    else
      inter_insert(location, pruned_list, parameters, 0);

    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::disable_delete(const Parameters &parameters,
                                     const bool consolidate) {
    if (!_can_delete) {
      std::cerr << "Delete not currently enabled" << std::endl;
      return -1;
    }
    if (!_enable_tags) {
      std::cerr << "Point tag array not instantiated" << std::endl;
      throw diskann::ANNException("Point tag array not instantiated", -1,
                                  __FUNCSIG__, __FILE__, __LINE__);
    }
    if (_eager_done) {
      if (_tag_to_location.size() != _nd) {
        std::cerr << "Tags to points array wrong sized : "
                  << _tag_to_location.size() << std::endl;
        return -2;
      }
    } else if (_tag_to_location.size() + _delete_set.size() != _nd) {
      std::cerr << "Tags to points array wrong sized"
                << "\n_tag_to_location.size():  " << _tag_to_location.size()
                << "\n_delete_set.size():  " << _delete_set.size()
                << "\n_nd:  " << _nd << std::endl;
      return -2;
    }

    if (_eager_done) {
      if (_location_to_tag.size() != _nd) {
        std::cerr << "Points to tags array wrong sized" << std::endl;
        return -3;
      }
    } else if (_location_to_tag.size() + _delete_set.size() != _nd) {
      std::cerr << "Points to tags array wrong sized "
                << _location_to_tag.size() << " " << _delete_set.size()
                << std::endl;
      return -3;
    }

    if (consolidate) {
      consolidate_deletes(parameters);
    }

    _can_delete = false;
    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::lazy_delete(const TagT &tag) {
    if ((_eager_done) && (!_data_compacted)) {
      std::cerr << "Eager delete requests were issued but data was not "
                   "compacted, cannot proceed with lazy_deletes"
                << std::endl;
      return -2;
    }
    _lazy_done = true;
    if (_tag_to_location.find(tag) == _tag_to_location.end()) {
      std::cerr << "Delete tag not found" << std::endl;
      return -1;
    }
    assert(_tag_to_location[tag] < _max_points);
    _delete_set.insert(_tag_to_location[tag]);
    _location_to_tag.erase(_tag_to_location[tag]);
    _tag_to_location.erase(tag);
    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::lazy_delete(const tsl::robin_set<TagT> &tags,
                                  std::vector<TagT> &failed_tags) {
    if (failed_tags.size() > 0) {
      std::cerr << "failed_tags should be passed as an empty list" << std::endl;
      return -3;
    }
    if ((_eager_done) && (!_data_compacted)) {
      std::cout << "Eager delete requests were issued but data was not "
                   "compacted, cannot proceed with lazy_deletes"
                << std::endl;
      return -2;
    }

    _lazy_done = true;

    for (auto tag : tags) {
      //      assert(_tag_to_location[tag] < _max_points);
      if (_tag_to_location.find(tag) == _tag_to_location.end()) {
        failed_tags.push_back(tag);
      } else {
        _delete_set.insert(_tag_to_location[tag]);
        _location_to_tag.erase(_tag_to_location[tag]);
        _tag_to_location.erase(tag);
      }
    }

    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::extract_data(
      T *ret_data, std::unordered_map<TagT, unsigned> &tag_to_location) {
    if (!_data_compacted) {
      std::cerr
          << "Error! Data not compacted. Cannot give access to private data."
          << std::endl;
      return -1;
    }
    std::memset(ret_data, 0, (size_t) _aligned_dim * _nd * sizeof(T));
    std::memcpy(ret_data, _data, (size_t)(_aligned_dim) *_nd * sizeof(T));
    tag_to_location = _tag_to_location;
    return 0;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_location_to_tag(
      std::unordered_map<unsigned, TagT> &ret_loc_to_tag) {
    ret_loc_to_tag = _location_to_tag;
  }

  /*  Internals of the library */
  // EXPORTS
  template DISKANN_DLLEXPORT class Index<float, int32_t>;
  template DISKANN_DLLEXPORT class Index<int8_t, int32_t>;
  template DISKANN_DLLEXPORT class Index<uint8_t, int32_t>;
  template DISKANN_DLLEXPORT class Index<float, uint32_t>;
  template DISKANN_DLLEXPORT class Index<int8_t, uint32_t>;
  template DISKANN_DLLEXPORT class Index<uint8_t, uint32_t>;
  template DISKANN_DLLEXPORT class Index<float, int64_t>;
  template DISKANN_DLLEXPORT class Index<int8_t, int64_t>;
  template DISKANN_DLLEXPORT class Index<uint8_t, int64_t>;
  template DISKANN_DLLEXPORT class Index<float, uint64_t>;
  template DISKANN_DLLEXPORT class Index<int8_t, uint64_t>;
  template DISKANN_DLLEXPORT class Index<uint8_t, uint64_t>;
}  // namespace diskann
