#include "pq_flash_index.h"
#include <malloc.h>
#include "percentile_stats.h"

#include <omp.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iterator>
#include <thread>
#include "distance.h"
#include "exceptions.h"
#include "parameters.h"
#include "timer.h"
#include "utils.h"

#include "tcmalloc/malloc_extension.h"
#include "tsl/robin_set.h"

#ifdef _WINDOWS
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#else
#include "linux_aligned_file_reader.h"
#include "mem_aligned_file_reader.h"
#endif

#define READ_U64(stream, val) stream.read((char *) &val, sizeof(_u64))
#define READ_UNSIGNED(stream, val) stream.read((char *) &val, sizeof(unsigned))

// sector # on disk where node_id is present
#define NODE_SECTOR_NO(node_id) (((_u64)(node_id)) / nnodes_per_sector + 1)

// obtains region of sector containing node
#define OFFSET_TO_NODE(sector_buf, node_id) \
  ((char *) sector_buf + (((_u64) node_id) % nnodes_per_sector) * max_node_len)

// offset into sector where node_id's nhood starts
#define NODE_SECTOR_OFFSET(sector_buf, node_id) \
  ((char *) sector_buf +                        \
   ((((_u64) node_id) % nnodes_per_sector) * max_node_len))

// returns region of `node_buf` containing [NNBRS][NBR_ID(_u32)]
#define OFFSET_TO_NODE_NHOOD(node_buf) \
  (unsigned *) ((char *) node_buf + data_dim * sizeof(T))

// returns region of `node_buf` containing [COORD(T)]
#define OFFSET_TO_NODE_COORDS(node_buf) (T *) (node_buf)

namespace {
  void aggregate_coords(const unsigned *ids, const _u64 n_ids,
                        const _u8 *all_coords, const _u64 ndims, _u8 *out) {
    for (_u64 i = 0; i < n_ids; i++) {
      memcpy(out + i * ndims, all_coords + ids[i] * ndims, ndims * sizeof(_u8));
    }
  }

  void pq_dist_lookup(const _u8 *pq_ids, const _u64 n_pts,
                      const _u64 pq_nchunks, const float *pq_dists,
                      float *dists_out) {
    _mm_prefetch((char *) dists_out, _MM_HINT_T0);
    _mm_prefetch((char *) pq_ids, _MM_HINT_T0);
    _mm_prefetch((char *) (pq_ids + 64), _MM_HINT_T0);
    _mm_prefetch((char *) (pq_ids + 128), _MM_HINT_T0);
    memset(dists_out, 0, n_pts * sizeof(float));
    for (_u64 chunk = 0; chunk < pq_nchunks; chunk++) {
      const float *chunk_dists = pq_dists + 256 * chunk;
      if (chunk < pq_nchunks - 1) {
        _mm_prefetch((char *) (chunk_dists + 256), _MM_HINT_T0);
      }
      for (_u64 idx = 0; idx < n_pts; idx++) {
        _u8 pq_centerid = pq_ids[pq_nchunks * idx + chunk];
        dists_out[idx] += chunk_dists[pq_centerid];
      }
    }
  }

  template<typename T>
  std::shared_ptr<diskann::Distance<T>> get_distance_function();

  template<>
  std::shared_ptr<diskann::Distance<float>> get_distance_function() {
    return std::shared_ptr<diskann::Distance<float>>(new diskann::DistanceL2());
  }
  template<>
  std::shared_ptr<diskann::Distance<uint8_t>> get_distance_function() {
    return std::shared_ptr<diskann::Distance<uint8_t>>(
        new diskann::DistanceL2UInt8());
  }
  template<>
  std::shared_ptr<diskann::Distance<int8_t>> get_distance_function() {
    return std::shared_ptr<diskann::Distance<int8_t>>(
        new diskann::DistanceL2Int8());
  }

}  // namespace

namespace diskann {
  template<typename T>
  DiskNode<T>::DiskNode(uint32_t id, T *coords, uint32_t *nhood) : id(id) {
    this->coords = coords;
    this->nnbrs = *nhood;
    this->nbrs = nhood + 1;
  }

  // structs for DiskNode
  template struct DiskNode<float>;
  template struct DiskNode<uint8_t>;
  template struct DiskNode<int8_t>;

  template<typename T, typename TagT>
  PQFlashIndex<T, TagT>::PQFlashIndex(
      std::shared_ptr<AlignedFileReader> &fileReader)
      : reader(fileReader) {
    this->dist_cmp = ::get_distance_function<T>();
    this->dist_cmp_float = ::get_distance_function<float>();
  }

  template<typename T, typename TagT>
  PQFlashIndex<T, TagT>::~PQFlashIndex() {
    if (data != nullptr) {
      delete[] data;
    }
    std::cout << "Thread Data size: " << this->thread_data.size() << "\n";
    assert(!this->thread_data.empty());

    if (centroid_data != nullptr)
      aligned_free(centroid_data);
    // delete backing bufs for nhood and coord cache
    if (nhood_cache_buf != nullptr) {
      delete[] nhood_cache_buf;
      diskann::aligned_free(coord_cache_buf);
    }

    if (load_flag) {
      this->destroy_thread_data();
      reader->close();
      // delete reader; //not deleting reader because it is now passed by ref.
    }
    if (medoids != nullptr)
      delete[] medoids;

    MallocExtension::instance()->ReleaseFreeMemory();
  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::setup_thread_data(_u64 nthreads) {
//    std::cout << "Setting up thread-specific contexts for nthreads: "
//              << nthreads << "\n";
// omp parallel for to generate unique thread IDs
#pragma omp parallel for
    for (_s64 thread = 0; thread < (_s64) nthreads; thread++) {
#pragma omp critical
      {
        this->reader->register_thread();
        IOContext       ctx = this->reader->get_ctx();
        QueryScratch<T> scratch;
        _u64 coord_alloc_size = ROUND_UP(MAX_N_CMPS * this->aligned_dim, 256);
        diskann::alloc_aligned((void **) &scratch.coord_scratch,
                               coord_alloc_size, 256);
        // scratch.coord_scratch = new T[MAX_N_CMPS * this->aligned_dim];
        // //Gopal. Commenting out the reallocation!
        diskann::alloc_aligned((void **) &scratch.sector_scratch,
                               MAX_N_SECTOR_READS * SECTOR_LEN, SECTOR_LEN);
        diskann::alloc_aligned((void **) &scratch.aligned_scratch,
                               256 * sizeof(float), 256);
        diskann::alloc_aligned((void **) &scratch.aligned_pq_coord_scratch,
                               32768 * 32 * sizeof(_u8), 256);
        diskann::alloc_aligned((void **) &scratch.aligned_pqtable_dist_scratch,
                               25600 * sizeof(float), 256);
        diskann::alloc_aligned((void **) &scratch.aligned_dist_scratch,
                               512 * sizeof(float), 256);
        diskann::alloc_aligned((void **) &scratch.aligned_query_T,
                               this->aligned_dim * sizeof(T), 8 * sizeof(T));
        diskann::alloc_aligned((void **) &scratch.aligned_query_float,
                               this->aligned_dim * sizeof(float),
                               8 * sizeof(float));

        memset(scratch.aligned_scratch, 0, 256 * sizeof(float));
        memset(scratch.coord_scratch, 0, coord_alloc_size);
        memset(scratch.aligned_query_T, 0, this->aligned_dim * sizeof(T));
        memset(scratch.aligned_query_float, 0,
               this->aligned_dim * sizeof(float));

        ThreadData<T> data;
        data.ctx = ctx;
        data.scratch = scratch;
        this->thread_data.push(data);
        this->thread_data_backing_buf.push_back(data);
      }
    }
    load_flag = true;
  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::destroy_thread_data() {
    //    std::cerr << "Clearing scratch" << std::endl;
    // assert(this->thread_data.size() == this->max_nthreads);
    for (auto &data : this->thread_data_backing_buf) {
      auto &scratch = data.scratch;
      diskann::aligned_free((void *) scratch.coord_scratch);
      diskann::aligned_free((void *) scratch.sector_scratch);
      diskann::aligned_free((void *) scratch.aligned_scratch);
      diskann::aligned_free((void *) scratch.aligned_pq_coord_scratch);
      diskann::aligned_free((void *) scratch.aligned_pqtable_dist_scratch);
      diskann::aligned_free((void *) scratch.aligned_dist_scratch);
      diskann::aligned_free((void *) scratch.aligned_query_float);
      diskann::aligned_free((void *) scratch.aligned_query_T);
    }
    this->reader->deregister_all_threads();
  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::load_cache_list(
      std::vector<uint32_t> &node_list) {
    //    std::cout << "Loading the cache list into memory.." << std::flush;
    _u64 num_cached_nodes = node_list.size();

    // borrow thread data
    ThreadData<T> this_thread_data = this->thread_data.pop();
    while (this_thread_data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      this_thread_data = this->thread_data.pop();
    }

    IOContext ctx = this_thread_data.ctx;

    nhood_cache_buf = new unsigned[num_cached_nodes * (max_degree + 1)];
    memset(nhood_cache_buf, 0, num_cached_nodes * (max_degree + 1));

    _u64 coord_cache_buf_len = num_cached_nodes * aligned_dim;
    diskann::alloc_aligned((void **) &coord_cache_buf,
                           coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
    memset(coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));

    size_t BLOCK_SIZE = 8;
    size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, BLOCK_SIZE);

    for (_u64 block = 0; block < num_blocks; block++) {
      _u64 start_idx = block * BLOCK_SIZE;
      _u64 end_idx = (std::min)(num_cached_nodes, (block + 1) * BLOCK_SIZE);
      std::vector<AlignedRead> read_reqs;
      std::vector<std::pair<_u32, char *>> nhoods;
      for (_u64 node_idx = start_idx; node_idx < end_idx; node_idx++) {
        AlignedRead read;
        char *      buf = nullptr;
        alloc_aligned((void **) &buf, SECTOR_LEN, SECTOR_LEN);
        nhoods.push_back(std::make_pair(node_list[node_idx], buf));
        read.len = SECTOR_LEN;
        read.buf = buf;
        read.offset = NODE_SECTOR_NO(node_list[node_idx]) * SECTOR_LEN;
        read_reqs.push_back(read);
      }

      reader->read(read_reqs, ctx);

      _u64 node_idx = start_idx;
      for (auto &nhood : nhoods) {
        char *node_buf = OFFSET_TO_NODE(nhood.second, nhood.first);
        T *   node_coords = OFFSET_TO_NODE_COORDS(node_buf);
        T *   cached_coords = coord_cache_buf + node_idx * aligned_dim;
        memcpy(cached_coords, node_coords, data_dim * sizeof(T));
        coord_cache.insert(std::make_pair(nhood.first, cached_coords));

        // insert node nhood into nhood_cache
        unsigned *node_nhood = OFFSET_TO_NODE_NHOOD(node_buf);

        auto      nnbrs = *node_nhood;
        unsigned *nbrs = node_nhood + 1;
        // std::cerr << "CACHE: nnbrs = " << nnbrs << "\n";
        std::pair<_u32, unsigned *> cnhood;
        cnhood.first = nnbrs;
        cnhood.second = nhood_cache_buf + node_idx * (max_degree + 1);
        memcpy(cnhood.second, nbrs, nnbrs * sizeof(unsigned));
        nhood_cache.insert(std::make_pair(nhood.first, cnhood));
        aligned_free(nhood.second);
        node_idx++;
      }
    }
    // return thread data
    this->thread_data.push(this_thread_data);
    //    std::cout << "..done." << std::endl;
  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::generate_cache_list_from_sample_queries(
      std::string sample_bin, _u64 l_search, _u64 beamwidth,
      _u64 num_nodes_to_cache, std::vector<uint32_t> &node_list) {
    this->count_visited_nodes = true;
    this->node_visit_counter.clear();
    this->node_visit_counter.resize(this->num_points);
    for (_u32 i = 0; i < node_visit_counter.size(); i++) {
      this->node_visit_counter[i].first = i;
      this->node_visit_counter[i].second = 0;
    }

    _u64 sample_num, sample_dim, sample_aligned_dim;
    T *  samples;

    if (file_exists(sample_bin)) {
      diskann::load_aligned_bin<T>(sample_bin, samples, sample_num, sample_dim,
                                   sample_aligned_dim);
    } else {
      std::cerr << "Sample bin file not found. Not generating cache."
                << std::endl;
      return;
    }

    std::vector<uint64_t> tmp_result_ids_64(sample_num, 0);
    std::vector<float>    tmp_result_dists(sample_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t) sample_num; i++) {
      cached_beam_search(samples + (i * sample_aligned_dim), 1, l_search,
                         tmp_result_ids_64.data() + (i * 1),
                         tmp_result_dists.data() + (i * 1), beamwidth);
    }

    std::sort(this->node_visit_counter.begin(), node_visit_counter.end(),
              [](std::pair<_u32, _u32> &left, std::pair<_u32, _u32> &right) {
                return left.second > right.second;
              });
    node_list.clear();
    node_list.shrink_to_fit();
    node_list.reserve(num_nodes_to_cache);
    for (_u64 i = 0; i < num_nodes_to_cache; i++) {
      node_list.push_back(this->node_visit_counter[i].first);
    }
    this->count_visited_nodes = false;

    diskann::aligned_free(samples);
  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::cache_bfs_levels(
      _u64 num_nodes_to_cache, std::vector<uint32_t> &node_list) {
    node_list.clear();

    // borrow thread data
    ThreadData<T> this_thread_data = this->thread_data.pop();
    while (this_thread_data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      this_thread_data = this->thread_data.pop();
    }

    IOContext ctx = this_thread_data.ctx;

    std::unique_ptr<tsl::robin_set<unsigned>> cur_level, prev_level;
    cur_level = std::make_unique<tsl::robin_set<unsigned>>();
    prev_level = std::make_unique<tsl::robin_set<unsigned>>();

    for (_u64 miter = 0; miter < num_medoids; miter++) {
      cur_level->insert(medoids[miter]);
    }

    _u64 lvl = 1;
    //    uint64_t prev_node_list_size = 0;
    while ((node_list.size() + cur_level->size() < num_nodes_to_cache) &&
           cur_level->size() != 0) {
      // swap prev_level and cur_level
      std::swap(prev_level, cur_level);
      // clear cur_level
      cur_level->clear();

      std::vector<unsigned> nodes_to_expand;

      for (const unsigned &id : *prev_level) {
        if (std::find(node_list.begin(), node_list.end(), id) !=
            node_list.end()) {
          continue;
        }
        node_list.push_back(id);
        nodes_to_expand.push_back(id);
      }

      std::random_shuffle(nodes_to_expand.begin(), nodes_to_expand.end());

      //      std::cout << "Level: " << lvl << std::flush;
      bool finish_flag = false;

      uint64_t BLOCK_SIZE = 1024;
      uint64_t nblocks = DIV_ROUND_UP(nodes_to_expand.size(), BLOCK_SIZE);
      for (size_t block = 0; block < nblocks && !finish_flag; block++) {
        //        std::cout << "." << std::flush;
        size_t start = block * BLOCK_SIZE;
        size_t end =
            (std::min)((block + 1) * BLOCK_SIZE, nodes_to_expand.size());
        std::vector<AlignedRead> read_reqs;
        std::vector<std::pair<_u32, char *>> nhoods;
        for (size_t cur_pt = start; cur_pt < end; cur_pt++) {
          char *buf = nullptr;
          alloc_aligned((void **) &buf, SECTOR_LEN, SECTOR_LEN);
          nhoods.push_back(std::make_pair(nodes_to_expand[cur_pt], buf));
          AlignedRead read;
          read.len = SECTOR_LEN;
          read.buf = buf;
          read.offset = NODE_SECTOR_NO(nodes_to_expand[cur_pt]) * SECTOR_LEN;
          read_reqs.push_back(read);
        }
        // issue read requests
        reader->read(read_reqs, ctx);
        // process each nhood buf
        for (auto &nhood : nhoods) {
          // insert node coord into coord_cache
          char *    node_buf = OFFSET_TO_NODE(nhood.second, nhood.first);
          unsigned *node_nhood = OFFSET_TO_NODE_NHOOD(node_buf);
          _u64      nnbrs = (_u64) *node_nhood;
          unsigned *nbrs = node_nhood + 1;
          // explore next level
          for (_u64 j = 0; j < nnbrs && !finish_flag; j++) {
            if (std::find(node_list.begin(), node_list.end(), nbrs[j]) ==
                node_list.end()) {
              cur_level->insert(nbrs[j]);
            }
            if (cur_level->size() + node_list.size() >= num_nodes_to_cache) {
              finish_flag = true;
            }
          }
          aligned_free(nhood.second);
        }
      }

      //      std::cout << ". #nodes: " << node_list.size() -
      //      prev_node_list_size
      //                << ", #nodes thus far: " << node_list.size() <<
      //                std::endl;
      //      prev_node_list_size = node_list.size();
      lvl++;
    }

    std::vector<uint32_t> cur_level_node_list;
    for (const unsigned &p : *cur_level)
      cur_level_node_list.push_back(p);

    std::random_shuffle(cur_level_node_list.begin(), cur_level_node_list.end());
    size_t residual = num_nodes_to_cache - node_list.size();

    for (size_t i = 0; i < (std::min)(residual, cur_level_node_list.size());
         i++)
      node_list.push_back(cur_level_node_list[i]);

    //    std::cout << "Level: " << lvl << std::flush;
    //    std::cout << ". #nodes: " << node_list.size() - prev_node_list_size
    //              << ", #nodes thus far: " << node_list.size() << std::endl;

    // return thread data
    this->thread_data.push(this_thread_data);
  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::use_medoids_data_as_centroids() {
    if (centroid_data != nullptr)
      aligned_free(centroid_data);
    alloc_aligned(((void **) &centroid_data),
                  num_medoids * aligned_dim * sizeof(float), 32);
    std::memset(centroid_data, 0, num_medoids * aligned_dim * sizeof(float));

    // borrow ctx
    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }
    IOContext ctx = data.ctx;
    //    std::cout << "Loading centroid data from medoids vector data of "
    //              << num_medoids << " medoid(s)" << std::endl;
    for (uint64_t cur_m = 0; cur_m < num_medoids; cur_m++) {
      auto medoid = medoids[cur_m];
      // read medoid nhood
      char *medoid_buf = nullptr;
      alloc_aligned((void **) &medoid_buf, SECTOR_LEN, SECTOR_LEN);
      std::vector<AlignedRead> medoid_read(1);
      medoid_read[0].len = SECTOR_LEN;
      medoid_read[0].buf = medoid_buf;
      medoid_read[0].offset = NODE_SECTOR_NO(medoid) * SECTOR_LEN;
      reader->read(medoid_read, ctx);

      // all data about medoid
      char *medoid_node_buf = OFFSET_TO_NODE(medoid_buf, medoid);

      // add medoid coords to `coord_cache`
      T *medoid_coords = new T[data_dim];
      T *medoid_disk_coords = OFFSET_TO_NODE_COORDS(medoid_node_buf);
      memcpy(medoid_coords, medoid_disk_coords, data_dim * sizeof(T));

      for (uint32_t i = 0; i < data_dim; i++)
        centroid_data[cur_m * aligned_dim + i] = medoid_coords[i];
      delete[] medoid_coords;

      aligned_free(medoid_buf);
    }

    // return ctx
    this->thread_data.push(data);
    this->thread_data.push_notify_all();
  }

  template<typename T, typename TagT>
  int PQFlashIndex<T, TagT>::load(uint32_t num_threads, const char *pq_prefix,
                                  const char *disk_index_file, bool load_tags) {
    std::string pq_table_bin = std::string(pq_prefix) + "_pivots.bin";
    std::string pq_compressed_vectors =
        std::string(pq_prefix) + "_compressed.bin";
    std::string medoids_file = std::string(disk_index_file) + "_medoids.bin";
    std::string centroids_file =
        std::string(disk_index_file) + "_centroids.bin";

    size_t pq_file_dim, pq_file_num_centroids;
    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim);

    this->disk_index_file = std::string(disk_index_file);

    if (pq_file_num_centroids != 256) {
      std::cout << "Error. Number of PQ centroids is not 256. Exitting."
                << std::endl;
      return -1;
    }

    this->data_dim = pq_file_dim;
    this->aligned_dim = ROUND_UP(pq_file_dim, 8);

    size_t npts_u64, nchunks_u64;
    diskann::load_bin<_u8>(pq_compressed_vectors, data, npts_u64, nchunks_u64);

    this->num_points = npts_u64;
    this->n_chunks = nchunks_u64;

    pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);

    /*
        _u64 npts_lrn, ndims_lrn, ndims_lrn_alg;

        std::string pq_deflated = pq_compressed_vectors + "_full.bin";
        float* deflated_pq;
        diskann::load_aligned_bin<float> (pq_deflated, deflated_pq, npts_lrn,
    ndims_lrn, ndims_lrn_alg);


        _u32 i1 = rand() % npts_lrn;

    //    std::vector<_u32> i2s;
        _u8* dest_vecs = new _u8[100*nchunks_u64];
        std::vector<float> dest_dists(100,0);
        std::vector<float> dest_dists_pq(100,0);
        _u32 counter=0;
        while (counter < 100) {

            _u32 idx = rand() % npts_lrn;
            dest_dists[counter] = dist_cmp_float->compare(deflated_pq +
    i1*ndims_lrn, deflated_pq + idx*ndims_lrn, ndims_lrn);
            std::memcpy(dest_vecs + counter*nchunks_u64, data + idx*nchunks_u64,
    nchunks_u64*sizeof(_u8));
            counter++;
        }

        _u8* src_vec = data + i1 * nchunks_u64;

        pq_table.compute_distances(src_vec, dest_vecs, dest_dists_pq.data(),
    100);

       for (_u32 i = 0; i < 100; i++) {
           std::cout<<dest_dists[i] << ", "<<dest_dists_pq[i] << " ";
       }
    */

    /*      std::string in_string = "/raid/data/SIFT1M/sift_learn.bin";
        std::string out_string = "/home/rakri/pq_learn.bin";

        float *sift_learn;
        _u64   npts_lrn, ndims_lrn;

        diskann::load_bin<float>(in_string, sift_learn, npts_lrn, ndims_lrn);

        _u8 *sift_lrn_out = new _u8[npts_lrn * nchunks_u64];

        std::cout << "learn npts, ndims :" << npts_lrn << ndims_lrn <<
       std::endl;
        for (_u32 i = 0; i < npts_lrn; i++) {
          pq_table.deflate_vec(sift_learn + i * ndims_lrn, sift_lrn_out + i *
       32);
        }

        diskann::save_bin<_u8>(out_string, sift_lrn_out, npts_lrn, nchunks_u64);

        std::cout << "Saved TMP PQ file";

        */

    //    std::cout
    //        << "Loaded PQ centroids and in-memory compressed vectors. #points:
    //        "
    //        << num_points << " #dim: " << data_dim
    //        << " #aligned_dim: " << aligned_dim << " #chunks: " << n_chunks
    //        << std::endl;

    // read nsg metadata
    std::ifstream nsg_meta(disk_index_file, std::ios::binary);

    size_t actual_index_size = get_file_size(disk_index_file);
    size_t expected_file_size;
    READ_U64(nsg_meta, expected_file_size);
    if (actual_index_size != expected_file_size) {
      std::cout << "File size mismatch for " << disk_index_file
                << " (size: " << actual_index_size << ")"
                << " with meta-data size: " << expected_file_size << std::endl;
      return -1;
    }

    _u64 disk_nnodes;
    READ_U64(nsg_meta, disk_nnodes);
    if (disk_nnodes != num_points) {
      std::cout << "Mismatch in #points for compressed data file and disk "
                   "index file: "
                << disk_nnodes << " vs " << num_points << std::endl;
      return -1;
    }

    size_t medoid_id_on_file;  // ndims;
    // read u64 to read ndims_64 of data points
    //    READ_U64(nsg_meta, ndims);
    READ_U64(nsg_meta, medoid_id_on_file);
    READ_U64(nsg_meta, max_node_len);
    READ_U64(nsg_meta, nnodes_per_sector);
    /*
        std::cout << "Disk num_points: " << disk_nnodes
                  << ", medoid_id_on_file: " << medoid_id_on_file
                  << ", max_node_len: " << max_node_len
                  << ", num_nodes_in_sector: " << nnodes_per_sector <<
       std::endl;
                  */

    max_degree = ((max_node_len - data_dim * sizeof(T)) / sizeof(unsigned)) - 1;
    /*
        std::cout << "Disk-Index File Meta-data: ";
        std::cout << "# nodes per sector: " << nnodes_per_sector;
        std::cout << ", max node len (bytes): " << max_node_len;
        std::cout << ", max node degree: " << max_degree << std::endl;
        */
    nsg_meta.close();

    // open AlignedFileReader handle to nsg_file
    std::string nsg_fname(disk_index_file);
    reader->open(nsg_fname, false, false);

    this->setup_thread_data(num_threads);
    this->max_nthreads = num_threads;

    if (file_exists(medoids_file)) {
      size_t tmp_dim;
      diskann::load_bin<uint32_t>(medoids_file, medoids, num_medoids, tmp_dim);

      if (tmp_dim != 1) {
        std::stringstream stream;
        stream << "Error loading medoids file. Expected bin format of m times "
                  "1 vector of uint32_t."
               << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }

      if (!file_exists(centroids_file)) {
        //        std::cout
        //            << "Centroid data file not found. Using corresponding
        //            vectors "
        //               "for the medoids "
        //            << std::endl;
        use_medoids_data_as_centroids();
      } else {
        size_t num_centroids, aligned_tmp_dim;
        diskann::load_aligned_bin<float>(centroids_file, centroid_data,
                                         num_centroids, tmp_dim,
                                         aligned_tmp_dim);
        if (aligned_tmp_dim != aligned_dim || num_centroids != num_medoids) {
          std::stringstream stream;
          stream << "Error loading centroids data file. Expected bin format of "
                    "m times data_dim vector of float, where m is number of "
                    "medoids "
                    "in medoids file."
                 << std::endl;
          throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                      __LINE__);
        }
      }
    } else {
      num_medoids = 1;
      medoids = new uint32_t[1];
      medoids[0] = (_u32)(medoid_id_on_file);
      use_medoids_data_as_centroids();
    }

    // load tags
    if (load_tags) {
      std::string tag_file = disk_index_file;
      tag_file = tag_file + ".tags";
      this->load_tags(tag_file);
    }
    //    std::cout << "done.." << std::endl;
    return 0;
  }

  template<typename T, typename TagT>
  _u64 PQFlashIndex<T, TagT>::return_nd() {
    return this->num_points;
  }

#ifdef USE_BING_INFRA
  bool getNextCompletedRequest(const IOContext &ctx, size_t size,
                               int &completedIndex) {
    bool waitsRemaining = false;
    for (int i = 0; i < size; i++) {
      auto ithStatus = (*ctx.m_pRequestsStatus)[i];
      if (ithStatus == IOContext::Status::READ_SUCCESS) {
        completedIndex = i;
        return true;
      } else if (ithStatus == IOContext::Status::READ_WAIT) {
        waitsRemaining = true;
      }
    }
    completedIndex = -1;
    return waitsRemaining;
  }
#endif

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::cached_beam_search(
      const T *query, const _u64 k_search, const _u64 l_search, _u64 *indices,
      float *distances, const _u64 beam_width, QueryStats *stats,
      Distance<T> *output_dist_func, TagT *res_tags) {
    // iterate to fixed point
    std::vector<Neighbor> expanded_nodes_info;
    this->disk_iterate_to_fixed_point(query, l_search, beam_width,
                                      expanded_nodes_info, nullptr,
                                      output_dist_func, stats);

    // fill in `indices`, `distances`
    for (uint32_t i = 0; i < k_search; i++) {
      indices[i] = expanded_nodes_info[i].id;
      if (distances != nullptr) {
        distances[i] = expanded_nodes_info[i].distance;
      }
      if (res_tags != nullptr && this->tags != nullptr) {
        res_tags[i] = this->tags[indices[i]];
      }
    }
  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::disk_iterate_to_fixed_point(
      const T *query1, const uint32_t l_search, const uint32_t beam_width,
      std::vector<Neighbor> &expanded_nodes_info,
      tsl::robin_map<uint32_t, T *> *coord_map, Distance<T> *output_dist_func,
      QueryStats *stats, ThreadData<T> *passthrough_data,
      tsl::robin_set<uint32_t> *exclude_nodes) {
    // only pull from sector scratch if ThreadData<T> not passed as arg
    ThreadData<T> data;
    if (passthrough_data == nullptr) {
      data = this->thread_data.pop();
      while (data.scratch.sector_scratch == nullptr) {
        this->thread_data.wait_for_push_notify();
        data = this->thread_data.pop();
      }
    } else {
      data = *passthrough_data;
    }

    /*    std::cout << "this->data_dim = " << this->data_dim << std::endl;
        for (uint32_t i = 0; i < this->data_dim; i++)
          std::cout << *(query1 + i) << " ;";
        std::cout << std::endl; */
    for (uint32_t i = 0; i < this->data_dim; i++) {
      data.scratch.aligned_query_float[i] = query1[i];
    }
    memcpy(data.scratch.aligned_query_T, query1, this->data_dim * sizeof(T));
    const T *    query = data.scratch.aligned_query_T;
    const float *query_float = data.scratch.aligned_query_float;

    IOContext ctx = data.ctx;
    auto      query_scratch = &(data.scratch);

    // reset query
    query_scratch->reset();

    // scratch space to compute distances between FP32 Query and INT8 data
    float *scratch = query_scratch->aligned_scratch;
    _mm_prefetch((char *) scratch, _MM_HINT_T0);

    // pointers to buffers for data
    T *   data_buf = query_scratch->coord_scratch;
    _u64 &data_buf_idx = query_scratch->coord_idx;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;

    // query <-> PQ chunk centers distances
    float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = query_scratch->aligned_dist_scratch;
    _u8 *  pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](
        const unsigned *ids, const _u64 n_ids, float *dists_out) {
      ::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                         pq_coord_scratch);
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                       dists_out);
    };

    Timer                 query_timer, io_timer, cpu_timer;
    std::vector<Neighbor> retset;
    retset.resize(l_search + 1);
    tsl::robin_set<_u64> visited(4096);

    // re-naming `expanded_nodes_info` to not change rest of the code
    std::vector<Neighbor> &full_retset = expanded_nodes_info;
    full_retset.reserve(4096);
    _u32                        best_medoid = 0;
    float                       best_dist = (std::numeric_limits<float>::max)();
    std::vector<SimpleNeighbor> medoid_dists;
    for (_u64 cur_m = 0; cur_m < num_medoids; cur_m++) {
      float cur_expanded_dist = dist_cmp_float->compare(
          query_float, centroid_data + aligned_dim * cur_m,
          (unsigned) aligned_dim);
      if (cur_expanded_dist < best_dist) {
        best_medoid = medoids[cur_m];
        best_dist = cur_expanded_dist;
      }
    }

    compute_dists(&best_medoid, 1, dist_scratch);
    retset[0].id = best_medoid;
    retset[0].distance = dist_scratch[0];
    retset[0].flag = true;
    visited.insert(best_medoid);

    unsigned cur_list_size = 1;

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned cmps = 0;
    unsigned hops = 0;
    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    std::vector<std::pair<unsigned, char *>> frontier_nhoods;
    std::vector<AlignedRead> frontier_read_reqs;
    std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
        cached_nhoods;

    while (k < cur_list_size) {
      auto nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      cached_nhoods.clear();
      sector_scratch_idx = 0;
      // find new beam
      // WAS: _u64 marker = k - 1;
      _u32 marker = k;
      _u32 num_seen = 0;
      while (marker < cur_list_size && frontier.size() < beam_width &&
             num_seen < beam_width) {
        if (retset[marker].flag) {
          num_seen++;
          auto iter = nhood_cache.find(retset[marker].id);
          if (iter != nhood_cache.end()) {
            cached_nhoods.push_back(
                std::make_pair(retset[marker].id, iter->second));
            if (stats != nullptr) {
              stats->n_cache_hits++;
            }
          } else {
            frontier.push_back(retset[marker].id);
          }
          retset[marker].flag = false;
          if (this->count_visited_nodes) {
            reinterpret_cast<std::atomic<_u32> &>(
                this->node_visit_counter[retset[marker].id].second)
                .fetch_add(1);
          }
        }
        marker++;
      }

      // read nhoods of frontier ids
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;
        for (_u64 i = 0; i < frontier.size(); i++) {
          auto id = frontier[i];
          std::pair<_u32, char *> fnhood;
          fnhood.first = id;
          fnhood.second = sector_scratch + sector_scratch_idx * SECTOR_LEN;
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          frontier_read_reqs.emplace_back(
              NODE_SECTOR_NO(((size_t) id)) * SECTOR_LEN, SECTOR_LEN,
              fnhood.second);
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }
        io_timer.reset();
#ifdef USE_BING_INFRA
        reader->read(frontier_read_reqs, ctx, true);  // async reader windows.
#else
        reader->read(frontier_read_reqs, ctx);  // synchronous IO linux
#endif
        if (stats != nullptr) {
          stats->io_us += io_timer.elapsed();
        }
      }

      // process cached nhoods
      for (auto &cached_nhood : cached_nhoods) {
        auto global_cache_iter = coord_cache.find(cached_nhood.first);
        T *  node_fp_coords = global_cache_iter->second;
        T *  node_fp_coords_copy = data_buf + (data_buf_idx * aligned_dim);
        data_buf_idx++;
        memcpy(node_fp_coords_copy, node_fp_coords, data_dim * sizeof(T));
        float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy,
                                                    (unsigned) aligned_dim);
        bool exclude_cur_node = false;
        if (exclude_nodes != nullptr) {
          exclude_cur_node =
              (exclude_nodes->find(cached_nhood.first) != exclude_nodes->end());
        }
        // only figure in final list if
        if (!exclude_cur_node) {
          // added for IndexMerger calls
          if (coord_map != nullptr) {
            coord_map->insert(
                std::make_pair(cached_nhood.first, node_fp_coords_copy));
          }
          full_retset.push_back(
              Neighbor((unsigned) cached_nhood.first, cur_expanded_dist, true));
        }
        _u64      nnbrs = cached_nhood.second.first;
        unsigned *node_nbrs = cached_nhood.second.second;

        // compute node_nbrs <-> query dists in PQ space
        cpu_timer.reset();
        compute_dists(node_nbrs, nnbrs, dist_scratch);
        if (stats != nullptr) {
          stats->n_cmps += nnbrs;
          stats->cpu_us += cpu_timer.elapsed();
        }

        // process prefetched nhood
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned id = node_nbrs[m];
          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            visited.insert(id);
            cmps++;
            float dist = dist_scratch[m];
            // std::cout << "cmp: " << id << ", dist: " << dist <<
            // std::endl; std::cerr << "dist: " << dist << std::endl;
            if (dist >= retset[cur_list_size - 1].distance &&
                (cur_list_size == l_search))
              continue;
            Neighbor nn(id, dist, true);
            auto     r = InsertIntoPool(
                retset.data(), cur_list_size,
                nn);  // Return position in sorted list where nn inserted.
            if (cur_list_size < l_search)
              ++cur_list_size;
            if (r < nk)
              nk = r;  // nk logs the best position in the retset that was
            // updated
            // due to neighbors of n.
          }
        }
      }
#ifdef USE_BING_INFRA
      // process each frontier nhood - compute distances to unvisited nodes
      int completedIndex = -1;
      // If we issued read requests and if a read is complete or there are reads
      // in wait
      // state, then enter the while loop.
      while (frontier_read_reqs.size() > 0 &&
             getNextCompletedRequest(ctx, frontier_read_reqs.size(),
                                     completedIndex)) {
        if (completedIndex == -1) {  // all reads are waiting
          continue;
        }
        auto &frontier_nhood = frontier_nhoods[completedIndex];
        (*ctx.m_pRequestsStatus)[completedIndex] = IOContext::PROCESS_COMPLETE;
#else
      for (auto &frontier_nhood : frontier_nhoods) {
#endif
        char *node_disk_buf =
            OFFSET_TO_NODE(frontier_nhood.second, frontier_nhood.first);
        unsigned *node_buf = OFFSET_TO_NODE_NHOOD(node_disk_buf);
        _u64      nnbrs = (_u64)(*node_buf);
        T *       node_fp_coords = OFFSET_TO_NODE_COORDS(node_disk_buf);
        assert(data_buf_idx < MAX_N_CMPS);

        T *node_fp_coords_copy = data_buf + (data_buf_idx * aligned_dim);
        data_buf_idx++;
        memcpy(node_fp_coords_copy, node_fp_coords, data_dim * sizeof(T));
        float cur_expanded_dist =
            dist_cmp->compare(query, node_fp_coords_copy, (unsigned) data_dim);
        bool exclude_cur_node = false;
        if (exclude_nodes != nullptr) {
          exclude_cur_node = (exclude_nodes->find(frontier_nhood.first) !=
                              exclude_nodes->end());
        }
        // if node is to be excluded from final search results
        if (!exclude_cur_node) {
          // added for IndexMerger calls
          if (coord_map != nullptr) {
            coord_map->insert(
                std::make_pair(frontier_nhood.first, node_fp_coords_copy));
          }
          full_retset.push_back(
              Neighbor(frontier_nhood.first, cur_expanded_dist, true));
        }
        unsigned *node_nbrs = (node_buf + 1);
        // compute node_nbrs <-> query dist in PQ space
        cpu_timer.reset();
        compute_dists(node_nbrs, nnbrs, dist_scratch);
        if (stats != nullptr) {
          stats->n_cmps += nnbrs;
          stats->cpu_us += cpu_timer.elapsed();
        }

        cpu_timer.reset();
        // process prefetch-ed nhood
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned id = node_nbrs[m];
          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            visited.insert(id);
            cmps++;
            float dist = dist_scratch[m];
            // std::cerr << "dist: " << dist << std::endl;
            if (stats != nullptr) {
              stats->n_cmps++;
            }
            if (dist >= retset[cur_list_size - 1].distance &&
                (cur_list_size == l_search))
              continue;
            Neighbor nn(id, dist, true);
            auto     r = InsertIntoPool(
                retset.data(), cur_list_size,
                nn);  // Return position in sorted list where nn inserted.
            if (cur_list_size < l_search)
              ++cur_list_size;
            if (r < nk)
              nk = r;  // nk logs the best position in the retset that was
                       // updated
                       // due to neighbors of n.
          }
        }

        if (stats != nullptr) {
          stats->cpu_us += cpu_timer.elapsed();
        }
      }

      // update best inserted position
      //

      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;

      hops++;
    }

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) {
                return left.distance < right.distance;
              });

    // return data to ConcurrentQueue only if popped from it
    if (passthrough_data == nullptr) {
      this->thread_data.push(data);
      this->thread_data.push_notify_all();
    }

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }
  //  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::compute_pq_dists(const T *query, const _u32 *ids,
                                               float *    fp_dists,
                                               const _u32 count) {
    // TODO (perf) :: more efficient impl without using populate_chunk_distances
    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }

    T *aligned_query = data.scratch.aligned_query_T;
    memcpy(aligned_query, query, this->data_dim * sizeof(T));
    auto query_scratch = &(data.scratch);

    // reset query
    query_scratch->reset();

    // query <-> PQ chunk centers distances
    float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(aligned_query, pq_dists);

    _u8 *pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](
        const unsigned *ids, const _u64 n_ids, float *dists_out) {
      ::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                         pq_coord_scratch);
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                       dists_out);
    };
    compute_dists(ids, count, fp_dists);

    // return scratch
    this->thread_data.push(data);
    this->thread_data.push_notify_all();
  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::compute_pq_dists(const _u32 src, const _u32 *ids,
                                               float *    fp_dists,
                                               const _u32 count,
                                               uint8_t *  aligned_scratch) {
    const _u8 *   src_ptr = this->data + (this->n_chunks * src);
    ThreadData<T> data;
    bool          popped = false;
    if (aligned_scratch == nullptr) {
      assert(false);
      // get buffer to store aggregated coords
      data = this->thread_data.pop();
      while (data.scratch.sector_scratch == nullptr) {
        this->thread_data.wait_for_push_notify();
        data = this->thread_data.pop();
      }
      popped = true;

      auto query_scratch = &(data.scratch);
      aligned_scratch = query_scratch->aligned_pq_coord_scratch;
    }

    // aggregate PQ coords into scratch
    ::aggregate_coords(ids, count, this->data, this->n_chunks, aligned_scratch);

    // compute distances
    this->pq_table.compute_distances(src_ptr, aligned_scratch, fp_dists, count);
    if (popped) {
      // return scratch
      this->thread_data.push(data);
      this->thread_data.push_notify_all();
    }
  }

  template<typename T, typename TagT>
  _u32 PQFlashIndex<T, TagT>::merge_read(std::vector<DiskNode<T>> &disk_nodes,
                                         _u32 &     start_id,
                                         const _u32 sector_count,
                                         char *     scratch) {
    assert(start_id % this->nnodes_per_sector == 0);
    assert(IS_ALIGNED(scratch, SECTOR_LEN));
    disk_nodes.clear();
    assert(scratch != nullptr);

    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }

    IOContext                ctx = data.ctx;
    std::vector<AlignedRead> read_req(1);
    _u64 start_off = NODE_SECTOR_NO(((size_t) start_id)) * SECTOR_LEN;
    _u64 n_sectors = ROUND_UP(this->num_points - start_id, nnodes_per_sector) /
                     nnodes_per_sector;
    n_sectors = std::min(n_sectors, (uint64_t) sector_count);
    assert(n_sectors > 0);
    read_req[0].buf = scratch;
    read_req[0].len = n_sectors * SECTOR_LEN;
    read_req[0].offset = start_off;

    // big sequential read
    this->reader->read(read_req, ctx);

    // create disk nodes
    _u32 cur_node_id = start_id;
    for (_u32 i = 0; i < n_sectors; i++) {
      char *sector_buf = scratch + (i * SECTOR_LEN);
      for (_u32 j = 0; j < nnodes_per_sector && cur_node_id < this->num_points;
           j++) {
        char *node_buf = OFFSET_TO_NODE(sector_buf, cur_node_id);
        disk_nodes.emplace_back(cur_node_id, OFFSET_TO_NODE_COORDS(node_buf),
                                OFFSET_TO_NODE_NHOOD(node_buf));
        cur_node_id++;
      }
    }

    // return scratch
    this->thread_data.push(data);
    this->thread_data.push_notify_all();

    // return cur_node_id as starting point for next iteration
    return cur_node_id;
  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::scan_deleted_nodes(
      const tsl::robin_set<uint32_t> &delete_set,
      std::vector<DiskNode<T>> &deleted_nodes, char *buf, char *backing_buf,
      const uint32_t sectors_per_scan) {
    assert(buf != nullptr);
    assert(IS_ALIGNED(buf, 4096));
    assert(IS_ALIGNED(backing_buf, 32));

    uint64_t backing_buf_unit_size = ROUND_UP(this->max_node_len, 32);
    uint64_t backing_buf_idx = 0;

    // TODO (perf) :: remove this memset
    memset(buf, 0, sectors_per_scan * SECTOR_LEN);

    // get ctx
    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }

    IOContext ctx = data.ctx;

    uint32_t                 n_scanned = 0;
    uint32_t                 base_offset = NODE_SECTOR_NO(0) * SECTOR_LEN;
    std::vector<AlignedRead> reads(1);
    reads[0].buf = buf;
    reads[0].len = sectors_per_scan * SECTOR_LEN;
    reads[0].offset = base_offset;
    while (n_scanned < this->num_points) {
      memset(buf, 0, sectors_per_scan * SECTOR_LEN);
      assert(this->reader);

      this->reader->read(reads, ctx);

      // scan each sector
      for (uint32_t i = 0; i < sectors_per_scan && n_scanned < this->num_points;
           i++) {
        char *sector_buf = buf + i * SECTOR_LEN;
        // scan each node
        for (uint32_t j = 0;
             j < nnodes_per_sector && n_scanned < this->num_points; j++) {
          char *node_buf = OFFSET_TO_NODE(sector_buf, n_scanned);
          // if in delete_set, add to deleted_nodes
          if (delete_set.find(n_scanned) != delete_set.end()) {
            char *buf_start =
                backing_buf + (backing_buf_idx * backing_buf_unit_size);
            backing_buf_idx++;
            memcpy(buf_start, node_buf, max_node_len);
            // create disk node object from backing buf instead of `buf`
            DiskNode<T> node(n_scanned, OFFSET_TO_NODE_COORDS(buf_start),
                             OFFSET_TO_NODE_NHOOD(buf_start));
            uint32_t nnbrs = node.nnbrs;
            assert(nnbrs < 512);
            assert(nnbrs > 0);
            deleted_nodes.push_back(node);
          }
          n_scanned++;
        }
      }
    }

    // return ctx
    this->thread_data.push(data);
    this->thread_data.push_notify_all();
  }

  template<typename T, typename TagT>
  std::vector<_u8> PQFlashIndex<T, TagT>::deflate_vector(const T *vec) {
    std::vector<_u8>   pq_coords(this->n_chunks);
    std::vector<float> fp_vec(this->data_dim);
    for (uint32_t i = 0; i < this->data_dim; i++) {
      fp_vec[i] = (float) vec[i];
    }
    this->pq_table.deflate_vec(fp_vec.data(), pq_coords.data());
    return pq_coords;
  }

  template<>
  std::vector<_u8> PQFlashIndex<float>::deflate_vector(const float *vec) {
    std::vector<_u8> pq_coords(this->n_chunks);
    this->pq_table.deflate_vec(vec, pq_coords.data());
    return pq_coords;
  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::reload_index(
      const std::string &disk_index_file,
      const std::string &pq_compressed_vectors, const uint32_t new_max_pts) {
    // reload PQ coords
    size_t npts_u64, nchunks_u64;
    delete this->data;

    std::cout << "RELOAD: Loading compressed vectors from "
              << pq_compressed_vectors << "\n";
    diskann::load_bin<_u8>(pq_compressed_vectors, data, npts_u64, nchunks_u64);
    assert(npts_u64 == new_max_pts);

    this->num_points = npts_u64;
    this->n_chunks = nchunks_u64;

    // close current FP
    this->reader->close();

    std::cout << "RELOAD: Loading graph from " << disk_index_file << "\n";
    // read new graph from disk
    std::ifstream nsg_meta(disk_index_file, std::ios::binary);

    size_t actual_index_size = get_file_size(disk_index_file);
    size_t expected_file_size;
    READ_U64(nsg_meta, expected_file_size);
    if (actual_index_size != expected_file_size) {
      std::cout << "File size mismatch for " << disk_index_file
                << " (size: " << actual_index_size << ")"
                << " with meta-data size: " << expected_file_size << std::endl;
      exit(-1);
    }

    _u64 disk_nnodes;
    READ_U64(nsg_meta, disk_nnodes);
    assert(disk_nnodes <= new_max_pts);

    size_t medoid_id_on_file;
    READ_U64(nsg_meta, medoid_id_on_file);
    std::cout << "Medoid-ID: " << medoid_id_on_file << "\n";
    this->medoids[0] = medoid_id_on_file;
    this->num_medoids = 1;
    READ_U64(nsg_meta, max_node_len);
    READ_U64(nsg_meta, nnodes_per_sector);
    max_degree = ((max_node_len - data_dim * sizeof(T)) / sizeof(unsigned)) - 1;
    nsg_meta.close();

    // point AlignedFileReader handle to nsg_file
    std::string nsg_fname(disk_index_file);
    reader->open(nsg_fname, true, false);
    // skip setup_thread_data() and other load() calls here

    // re-load tags
    std::string tag_file = disk_index_file;
    tag_file = tag_file + ".tags";
    this->load_tags(tag_file);

    // number of nodes to cache
    uint32_t node_cache_count =
        std::min(this->nhood_cache.size(), this->coord_cache.size());

    // clear cache
    std::cout << "RELOAD: Clearing cache.\n";
    delete[] this->nhood_cache_buf;
    this->nhood_cache.clear();
    delete[] this->coord_cache_buf;
    this->coord_cache.clear();

    // refresh cache
    std::cout << "RELOAD: Caching " << node_cache_count << " nodes.\n";
    std::vector<uint32_t> node_list;
    this->cache_bfs_levels(node_cache_count, node_list);
    this->load_cache_list(node_list);
  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::load_tags(const std::string &tag_file_name) {
    std::cout << "Loading tags from " << tag_file_name << "\n";
    /*     std::ifstream tag_file;
        tag_file = std::ifstream(tag_file_name);
        if (!tag_file.is_open()) {
          std::cerr << "PQFlashIndex:: Tag file not found at " << tag_file_name
                    << std::endl;
          // if failed to load tags, init with [0, num_points-1]
          this->tags.resize(this->num_points);
          TagT low = (TagT) 0;
          std::iota(this->tags.begin(), this->tags.end(), low);
          return;
        }
        // remove any state
        this->tags.clear();
        this->tags.reserve(this->num_points);

        unsigned id = 0;
        TagT     tag;
        while (tag_file >> tag) {
          this->tags.push_back(tag);
        }
        //    std::cout << "Loaded " << this->tags.size() << " tags.\n";
        tag_file.close();
    */
    size_t tag_num, tag_dim;
/*    TagT * tag_data;
    diskann::load_bin<TagT>(tag_file_name, tag_data, tag_num, tag_dim);
    this->tags.clear();
    this->tags.reserve(this->num_points);
    for (size_t i = 0; i < tag_num; i++)
      this->tags.push_back(*(tag_data + i));
    delete[] tag_data;
    */
    diskann::load_bin<TagT>(tag_file_name, tags, tag_num, tag_dim);
  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::passthrough_write(char *buf,
                                                const uint64_t offset,
                                                const uint64_t size) {
    // get ctx
    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }

    IOContext   ctx = data.ctx;
    AlignedRead write_req;
    write_req.buf = buf;
    write_req.len = size;
    write_req.offset = offset;

    this->reader->sequential_write(write_req, ctx);

    // return ctx
    this->thread_data.push(data);
    this->thread_data.push_notify_all();
  }

  // instantiations
  template class PQFlashIndex<float, int32_t>;
  template class PQFlashIndex<_s8, int32_t>;
  template class PQFlashIndex<_u8, int32_t>;
  template class PQFlashIndex<float, uint32_t>;
  template class PQFlashIndex<_s8, uint32_t>;
  template class PQFlashIndex<_u8, uint32_t>;
  template class PQFlashIndex<float, int64_t>;
  template class PQFlashIndex<_s8, int64_t>;
  template class PQFlashIndex<_u8, int64_t>;
  template class PQFlashIndex<float, uint64_t>;
  template class PQFlashIndex<_s8, uint64_t>;
  template class PQFlashIndex<_u8, uint64_t>;
}  // namespace diskann
