#include "neighbor.h"
#include "timer.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include "v2/index_merger.h"
#include "mem_aligned_file_reader.h"
#include <algorithm>
#include <cassert>
#include <csignal>
#include <iterator>
#include <mutex>
#include <thread>
#include <vector>
#include <limits>
#include <omp.h>

#include "tcmalloc/malloc_extension.h"

#define SECTORS_PER_MERGE (uint64_t)65536
// max number of points per mem index being merged -- 32M
#define MAX_PTS_PER_MEM_INDEX (uint64_t)(1 << 25)
#define INDEX_OFFSET (uint64_t) (MAX_PTS_PER_MEM_INDEX * 4)
#define MAX_INSERT_THREADS (uint64_t) 40
#define MAX_N_THREADS (uint64_t) 40
#define PER_THREAD_BUF_SIZE (uint64_t) (65536 * 64 * 4)

std::string TMP_FOLDER;
#define PQ_FLASH_INDEX_CACHE_SIZE 200000

namespace diskann {
  template<typename T, typename TagT>
  IndexMerger<T, TagT>::IndexMerger(const char* disk_in, const std::vector<std::string> &mem_in, const char* disk_out, 
                           const char* deleted_tags_file, const uint32_t ndims, Distance<T>* dist, const uint32_t beam_width,
                           const uint32_t range, const uint32_t l_index, const float alpha, const uint32_t maxc) {
	  std::cout << l_index << " , " << range << " , " << alpha << " , " << maxc << " , " << beam_width << std::endl;
    // book keeping
    this->ndims = ndims;
    this->aligned_ndims = ROUND_UP(this->ndims, 8);
    this->range = range;
    this->l_index = l_index;
    this->beam_width = beam_width;
    this->maxc = maxc;
    this->alpha = alpha;
    this->dist_cmp = dist;

    // load disk index
    this->pq_coords_file = std::string(disk_in) + "_pq_compressed.bin";
    this->disk_index_out_path = disk_out;
    this->disk_index_in_path = disk_in;
    std::shared_ptr<AlignedFileReader> reader = std::make_shared<LinuxAlignedFileReader>();
//    std::shared_ptr<AlignedFileReader> reader = std::make_shared<MemAlignedFileReader>();
    this->disk_index = new PQFlashIndex<T, TagT>(reader);
    std::cout << "Created PQFlashIndex inside index_merger " << std::endl;
    std::string pq_prefix = std::string(disk_in) + "_pq";
    std::string disk_index_file = std::string(disk_in) + "_disk.index";
    this->disk_index->load(MAX_N_THREADS, pq_prefix.c_str(), disk_index_file.c_str(), true);
    std::cout << "Loaded PQFlashIndex" << std::endl;
    std::vector<uint32_t> cache_node_list;
    this->disk_index->cache_bfs_levels(PQ_FLASH_INDEX_CACHE_SIZE, cache_node_list);
    this->disk_index->load_cache_list(cache_node_list);
    this->disk_tags = this->disk_index->get_tags();
    this->init_ids = this->disk_index->get_init_ids();
    this->disk_npts = this->disk_index->return_nd();
    this->disk_thread_data = this->disk_index->get_thread_data();
    auto res_pq = this->disk_index->get_pq_config();
    this->pq_data = res_pq.first;
    this->pq_nchunks = res_pq.second;
    this->nnodes_per_sector = this->disk_index->nnodes_per_sector;
    this->max_node_len = this->disk_index->max_node_len;
    
    // create deltas
    this->disk_delta = new GraphDelta(0, this->disk_npts);
    uint32_t base_offset = ROUND_UP(this->disk_npts, INDEX_OFFSET);

    // load mem-indices
    for(auto &mem_index_path : mem_in) {
      std::string ind_path = mem_index_path;
      std::string data_path = mem_index_path + ".data";
      std::ifstream bin_reader(data_path, std::ios::binary);
      uint32_t bin_npts, bin_ndims;
      bin_reader.read((char*) &bin_npts, sizeof(uint32_t));
      bin_reader.read((char*) &bin_ndims, sizeof(uint32_t));
      bin_reader.close();
      std::cout << "Index Path: " << ind_path << "\n";
      std::cout << "Data Path: " << data_path << "\n";
      std::cout << "Detected # pts = " << bin_npts << ", # dims = " << bin_ndims << "\n";
      // TODO (correct) :: take frozen points into account
  //    Index<T, TagT>* mem_index = new Index<T, TagT>(Metric::L2, bin_ndims, bin_npts + 100, 1, true);
      Index<T, TagT>* mem_index = new Index<T, TagT>(Metric::L2, bin_ndims, bin_npts + 100, 0, true);
      // TODO (verify) :: correct paths from one index path
//      mem_index->load(ind_path.c_str(), data_path.c_str(), true);
      _u64 n1, n2, n3;
      diskann::load_aligned_bin<T>(data_path, _data_load, n1, n2, n3);
//      this->mem_indices.push_back(mem_index);
      uint32_t npts = n1;
      assert(npts < MAX_PTS_PER_MEM_INDEX);
      this->mem_npts.push_back(npts);
//      this->mem_graphs.push_back(mem_index->get_graph());
//      this->mem_data.push_back(mem_index->get_data());
      this->mem_data.push_back(_data_load);
      uint32_t index_offset = base_offset;
      base_offset += INDEX_OFFSET;
      this->offset_ids.push_back(index_offset);
      this->mem_deltas.push_back(new GraphDelta(index_offset, npts));
      // manage tags
      std::unique_ptr<TagT[]> index_tags;
      index_tags.reset(new TagT[npts]);

      mem_index->load_tag_bin(mem_index_path + ".tags");
      const std::unordered_map<uint32_t, TagT> &loc_tag_map = *mem_index->get_tags();
      for(uint32_t k=0; k < npts; k++) {
        auto iter = loc_tag_map.find(k);
        if (iter == loc_tag_map.end()) {
          std::cout << "Index # " << this->mem_data.size() << " : missing tag for node #" << k << "\n";
	  exit(-1);
          index_tags[k] = (TagT) k;
        } else {
          index_tags[k] = iter->second;
        }
      }
      this->mem_tags.push_back(std::move(index_tags));
    }

/*    std::ifstream deleted_tag_reader(deleted_tags_file);
    TagT tag;
    while(deleted_tag_reader >> tag) {
      this->deleted_tags.insert(tag);
    }
    deleted_tag_reader.close();
    */
    std::cout << "Reading deleted tag list from " << deleted_tags_file << "\n";
    size_t tag_num, tag_dim;
    TagT * tag_data;
    diskann::load_bin<TagT>(deleted_tags_file, tag_data, tag_num, tag_dim);
    std::cout << "Detected " << tag_num << " tags to delete." << std::endl;
    for(size_t i = 0; i < tag_num; i++)
    {
        this->deleted_tags.insert(*(tag_data + i));
    }

    delete[] tag_data;
    std::cout << "Allocating thread scratch space -- " << PER_THREAD_BUF_SIZE / (1<<20) << " MB / thread.\n";
    alloc_aligned((void**) &this->thread_pq_scratch, MAX_N_THREADS * PER_THREAD_BUF_SIZE, SECTOR_LEN);
    this->thread_bufs.resize(MAX_N_THREADS);
    for(uint32_t i=0; i < thread_bufs.size(); i++) {
      this->thread_bufs[i] = this->thread_pq_scratch + i * PER_THREAD_BUF_SIZE;
    }
  }

  template<typename T, typename TagT>
  IndexMerger<T, TagT>::~IndexMerger() {
    // release scratch alloc memory
    // delete this->fp_alloc;
    // delete this->pq_alloc;

    // if (this->disk_index != nullptr)
    //   delete this->disk_index;
    delete this->disk_delta;
    for(auto &delta : this->mem_deltas) {
      delete delta;
    }
/*    for(auto &mem_index : this->mem_indices) {
      delete mem_index;
    } */
    aligned_free((void*) this->thread_pq_scratch);
    for(auto &data : this->mem_data)
    {
	delete [] data;
    }
  }

  template<typename T, typename TagT>
  void IndexMerger<T, TagT>::process_inserts_pq() {
      Timer total_insert_timer;
    this->insert_times.resize(MAX_N_THREADS, 0.0);
    this->delta_times.resize(MAX_N_THREADS, 0.0);
    // iterate through each vector in each mem index
    for(uint32_t i=0;i<this->mem_data.size(); i++) {
      std::cout << "Processing pq of inserts from mem-DiskANN #" << i + 1 << "\n";
      const tsl::robin_set<uint32_t> &deleted_set = this->mem_deleted_ids[i];
//      const T* coords = this->mem_data[i];
      const T* coords = _data_load;
      const uint32_t offset = this->offset_ids[i];
      const uint32_t count = this->mem_npts[i];
    // TODO (perf) :: trivially parallelizes ??
    #pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_N_THREADS)
      // iteratively insert each point into full index
      for(uint32_t j=0;j < count; j++) {
        // filter out -- `j` is deleted
        if (deleted_set.find(j) != deleted_set.end()) {
          continue;
        }

        // data for jth point
        const T* j_coords = coords + ((uint64_t)(this->aligned_ndims) * (uint64_t)j);
        const uint32_t j_id = offset + j;

        // get renamed ID
        const uint32_t j_renamed = this->rename(j_id);
        assert(j_renamed != std::numeric_limits<uint32_t>::max());
        
        // compute PQ coords
        std::vector<uint8_t> j_pq_coords = this->disk_index->deflate_vector(j_coords);
//        std::vector<uint8_t> j_pq_coords(this->pq_nchunks,0);

        // directly copy into PQFlashIndex PQ data
        const uint64_t j_pq_offset = (uint64_t) j_renamed * (uint64_t) this->pq_nchunks;
        memcpy(this->pq_data + j_pq_offset, j_pq_coords.data(), this->pq_nchunks * sizeof(uint8_t));
      }
    }

    std::cout << "Finished deflating all points\n";
    double e2e_time = ((double)total_insert_timer.elapsed())/(1000000.0);
    double insert_time = std::accumulate(this->insert_times.begin(), this->insert_times.end(), 0.0);
    double delta_time = std::accumulate(this->delta_times.begin(), this->delta_times.end(), 0.0);
    uint32_t n_inserts = std::accumulate(this->mem_npts.begin(), this->mem_npts.end(), 0);
    std::cout << "TIMER:: PQ time per point = " << insert_time / n_inserts << ", Delta = " << delta_time / n_inserts << "\n";
    std::cout <<" E2E pq time: "<< e2e_time << " sec" << std::endl;
  }

  template<typename T, typename TagT>
  void IndexMerger<T, TagT>::process_inserts() {
      Timer total_insert_timer;
    this->insert_times.resize(MAX_INSERT_THREADS, 0.0);
    this->delta_times.resize(MAX_INSERT_THREADS, 0.0);
    // iterate through each vector in each mem index
    for(uint32_t i=0;i<this->mem_data.size(); i++) {
      std::cout << "Processing inserts from mem-DiskANN #" << i + 1 << "\n";
      const tsl::robin_set<uint32_t> &deleted_set = this->mem_deleted_ids[i];
//      const T* coords = this->mem_data[i];
      const T* coords = _data_load;
      const uint32_t offset = this->offset_ids[i];
      const uint32_t count = this->mem_npts[i];

      size_t cur_cache_size = 0;
      MallocExtension::instance()->GetNumericProperty("tcmalloc.max_total_thread_cache_bytes", &cur_cache_size);
      std::cout << "Current cache size : " << (cur_cache_size >> 10) << " KiB\n" << std::endl;
      MallocExtension::instance()->SetNumericProperty("tcmalloc.max_total_thread_cache_bytes", 128 * 1024 * 1024);

    #pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_INSERT_THREADS)
      // iteratively insert each point into full index
      for(uint32_t j=0;j < count; j++) {
        // filter out -- `j` is deleted
        if (deleted_set.find(j) != deleted_set.end()) {
          continue;
        }

	      if(((j % 1000000) == 0) && (j > 0))
		      std::cout << "Finished inserting " << j << " points" << std::endl;
        // data for jth point
        const T* j_coords = coords + ((uint64_t)(this->aligned_ndims) * (uint64_t)j);
        const uint32_t j_id = offset + j;

        // insert into index
        this->insert_mem_vec(j_coords, j_id);
      }
    }

    std::cout << "Finished inserting all points\n";
    double e2e_time = ((double)total_insert_timer.elapsed())/(1000000.0);
    double insert_time = std::accumulate(this->insert_times.begin(), this->insert_times.end(), 0.0);
    double delta_time = std::accumulate(this->delta_times.begin(), this->delta_times.end(), 0.0);
    uint32_t n_inserts = std::accumulate(this->mem_npts.begin(), this->mem_npts.end(), 0);
    std::cout << "TIMER:: Insert time per point = " << insert_time / n_inserts << ", Delta = " << delta_time / n_inserts << "\n";
    std::cout <<" E2E insert time: "<< e2e_time << " sec" << std::endl;
  }

  template<typename T, typename TagT>
  void IndexMerger<T, TagT>::insert_mem_vec(const T* mem_vec, const uint32_t offset_id) {
	  Timer timer;
	 float insert_time, delta_time;
    // START: mem_vec has no ID, no presence in system
    std::vector<Neighbor>    pool;
    std::vector<Neighbor>    tmp;
    tsl::robin_map<uint32_t, T*> coord_map;

    // search on combined graph
    this->offset_iterate_to_fixed_point(mem_vec, offset_id, this->l_index, pool, coord_map);
    insert_time = timer.elapsed();

    // prune neighbors using alpha
    std::vector<uint32_t> new_nhood;
    prune_neighbors(mem_vec, coord_map, pool, new_nhood);

    // insert into PQ Flash Index
    this->disk_delta->insert_vector(offset_id, new_nhood.data(), new_nhood.size());
    this->disk_delta->inter_insert(offset_id, new_nhood.data(), new_nhood.size());
  
    // insert into graph
    for(auto &delta : this->mem_deltas) {
      delta->insert_vector(offset_id, new_nhood.data(), new_nhood.size());
     delta->inter_insert(offset_id, new_nhood.data(), new_nhood.size());
    }
    delta_time = timer.elapsed();
    // END: mem_vec now connected with new ID
    uint32_t thread_no = omp_get_thread_num();
    this->insert_times[thread_no] += insert_time;
    this->delta_times[thread_no] += delta_time;
  }

  template<typename T, typename TagT>
  void IndexMerger<T, TagT>::offset_iterate_to_fixed_point(const T* vec, const uint32_t offset_id, const uint32_t Lsize,
                                          std::vector<Neighbor> &expanded_nodes_info, 
                                          tsl::robin_map<uint32_t, T*> &coord_map) {
    std::vector<Neighbor> exp_node_info;
    exp_node_info.reserve(2 * Lsize);
    tsl::robin_map<uint32_t, T*> cmap;
    // first hit PQ iterate to fixed point
    // NOTE :: handling deletes for disk-index inside this call
    // this->disk_iterate_to_fixed_point(vec, this->l_index, exp_node_info, exp_node_id, best_l_nodes, cmap);
    uint32_t omp_thread_no = omp_get_thread_num();
    ThreadData<T> &thread_data = this->disk_thread_data[omp_thread_no];
//    ThreadData<T> * thread_data = nullptr;

    cmap.reserve(2 * Lsize);
    this->disk_index->disk_iterate_to_fixed_point(vec, Lsize, this->beam_width, exp_node_info, &cmap, nullptr, nullptr, &thread_data, &this->disk_deleted_ids);
    // std::cout << "Disk nodes: (" << best_l_nodes[0].id << ", " << best_l_nodes[0].distance;
    // std::cout << ") <-> : (" << best_l_nodes.back().id << ", " << best_l_nodes.back().distance << ")\n";

    // hit each mem index
    // NOTE :: handling deletes for mem-indices here
/*     for(uint32_t i=0;i<this->mem_indices.size();i++) {
      std::vector<Neighbor> exp_node_info_local;
      tsl::robin_map<uint32_t, T*> cmap_local;
      uint32_t index_offset = this->offset_ids[i];
      Index<T, TagT> *index = this->mem_indices[i];
      const tsl::robin_set<uint32_t> &deleted_set = this->mem_deleted_ids[i];
//      index->iterate_to_fixed_point(vec, this->l_index, exp_node_info_local, cmap_local, false);
      index->iterate_to_fixed_point(vec, this->l_index, exp_node_info_local, cmap_local, true);
      // std::cout << "Mem #" << i+1 << " nodes : (" << best_l_nodes_local[0].id << ", " << best_l_nodes_local[0].distance;
      // std::cout << ") <-> : (" << best_l_nodes_local.back().id << ", " << best_l_nodes_local.back().distance << ")\n";

      for(auto &nbr : exp_node_info_local) {
        uint32_t oid = nbr.id + index_offset;
        // filter out -- (1) offset_id, and (2) deleted ids
        if(deleted_set.find(nbr.id) != deleted_set.end() || oid == offset_id) {
          continue;
        }
        exp_node_info.emplace_back(nbr.id + index_offset, nbr.distance, nbr.flag);
      }
      for(auto &k_v : cmap_local) {
        uint32_t oid = k_v.first + index_offset;
        // filter out -- (1) offset_id, and (2) deleted ids
        if(deleted_set.find(k_v.first) != deleted_set.end() || oid == offset_id) {
          continue;
        }
        cmap.insert(std::make_pair(oid, k_v.second));
      }
    }
*/
    // reduce and pick top maxc expanded nodes only
    std::sort(exp_node_info.begin(), exp_node_info.end());
//    expanded_nodes_info.clear();
    expanded_nodes_info.reserve(this->maxc);
    expanded_nodes_info.insert(expanded_nodes_info.end(), exp_node_info.begin(), exp_node_info.end());
    
    // insert only relevant coords into coord_map
    for(auto &nbr : expanded_nodes_info) {
      uint32_t id = nbr.id;
      auto iter = cmap.find(id);
      assert(iter != cmap.end());
      coord_map.insert(std::make_pair(iter->first, iter->second));
    }
  }

  template<typename T, typename TagT>
  void IndexMerger<T, TagT>::prune_neighbors(const T* vec, const tsl::robin_map<uint32_t, T*> &coord_map, 
                                    std::vector<Neighbor> &pool, std::vector<uint32_t> &pruned_list) {
    if (pool.size() == 0)
      return;

    // sort the pool based on distance to query
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    result.reserve(range);
    std::vector<float> occlude_factor(pool.size(), 0);

    occlude_list(pool, vec, coord_map, result, occlude_factor);

    pruned_list.clear();
    assert(result.size() <= range);
    for (auto iter : result) {
      pruned_list.emplace_back(iter.id);
    }

    if (alpha > 1) {
      for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
        if (std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) == pruned_list.end())
          pruned_list.emplace_back(pool[i].id);
      }
    }
  }

  template<typename T, typename TagT>
  void IndexMerger<T, TagT>::prune_neighbors_pq(const uint32_t id, std::vector<Neighbor> &pool, 
                                                std::vector<uint32_t> &pruned_list, uint8_t* scratch) {
    if (pool.size() == 0)
      return;

    // sort the pool based on distance to query
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    result.reserve(this->range);
    std::vector<float> occlude_factor(pool.size(), 0);

    occlude_list_pq(pool, id, result, occlude_factor, scratch);

    pruned_list.clear();
    assert(result.size() <= range);
    for (auto iter : result) {
      pruned_list.emplace_back(iter.id);
    }

    if (alpha > 1) {
      for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
        if (std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) == pruned_list.end())
          pruned_list.emplace_back(pool[i].id);
      }
    }
  }

  template<typename T, typename TagT>
  void IndexMerger<T, TagT>::occlude_list(std::vector<Neighbor> &pool, const T *vec, 
                                    const tsl::robin_map<uint32_t, T*> &coord_map, 
                                    std::vector<Neighbor> &result, std::vector<float> &occlude_factor) {
    if (pool.empty())
      return;
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(!pool.empty());

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < range) {
      uint32_t start = 0;
      while (result.size() < range && (start) < pool.size() && start < maxc) {
        auto &p = pool[start];
        if (occlude_factor[start] > cur_alpha) {
          start++;
          continue;
        }
        occlude_factor[start] = std::numeric_limits<float>::max();
        result.push_back(p);
        for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
          if (occlude_factor[t] > alpha)
            continue;
          auto iter_right = coord_map.find(p.id);
          auto iter_left = coord_map.find(pool[t].id);
          // HAS to be in coord_map since it was expanded during iterate_to_fixed_point
          assert(iter_right != coord_map.end());
          assert(iter_left != coord_map.end());
          // WARNING :: correct, but not fast -- NO SIMD version if using MSVC, g++ should auto vectorize
          float djk = this->dist_cmp->compare(iter_left->second, iter_right->second, this->ndims);
          occlude_factor[t] = (std::max)(occlude_factor[t], pool[t].distance / djk);
        }
        start++;
      }
      cur_alpha *= 1.2;
    }
  }

  template<typename T, typename TagT>
void IndexMerger<T, TagT>::occlude_list_pq(std::vector<Neighbor> &pool, const uint32_t id, 
                                           std::vector<Neighbor> &result, std::vector<float> &occlude_factor, uint8_t* scratch) {
    if (pool.empty())
      return;
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(!pool.empty());

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < range) {
      uint32_t start = 0;
      while (result.size() < range && (start) < pool.size() && start < maxc) {
        auto &p = pool[start];
        if (occlude_factor[start] > cur_alpha) {
          start++;
          continue;
        }
        occlude_factor[start] = std::numeric_limits<float>::max();
        result.push_back(p);
        for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
          if (occlude_factor[t] > alpha)
            continue;
          // djk = dist(p.id, pool[t.id])
          float djk;
          this->disk_index->compute_pq_dists(p.id, &(pool[t].id), &djk, 1, scratch);
          occlude_factor[t] = (std::max)(occlude_factor[t], pool[t].distance / djk);
        }
        start++;
      }
      cur_alpha *= 1.2;
    }
  }

  template<typename T, typename TagT>
  void IndexMerger<T, TagT>::dump_to_disk(const std::vector<DiskNode<T>> &disk_nodes, const uint32_t start_id, 
                                 const char* buf, const uint32_t n_sectors) {
      assert(start_id % this->nnodes_per_sector == 0);
      uint32_t start_sector = (start_id / this->nnodes_per_sector) + 1;
      uint64_t start_off = start_sector * (uint64_t) SECTOR_LEN;

      // seek fp
      this->output_writer.seekp(start_off, std::ios::beg);

      // dump
      this->output_writer.write(buf, (uint64_t) n_sectors * (uint64_t) SECTOR_LEN);

      uint64_t nb_written = (uint64_t) this->output_writer.tellp() - (uint64_t) start_off;
      assert(nb_written == (uint64_t) n_sectors * (uint64_t) SECTOR_LEN);
  }

  template<typename T, typename TagT>
  void IndexMerger<T, TagT>::compute_deleted_ids() {
    // process disk deleted tags
    for(uint32_t i=0; i < this->disk_npts; i++) {
      TagT i_tag = this->disk_tags[i];
      if (this->deleted_tags.find(i_tag) != this->deleted_tags.end()) {
        this->disk_deleted_ids.insert(i);
      }
    }
    std::cout << "Found " << this->disk_deleted_ids.size() << " tags to delete from SSD-DiskANN\n";

    this->mem_deleted_ids.resize(this->mem_data.size());
    for(uint32_t i=0;i < this->mem_data.size(); i++) {
      tsl::robin_set<uint32_t> &deleted_ids = this->mem_deleted_ids[i];
      for(uint32_t id = 0; id < this->mem_npts[i]; id++) {
        const TagT tag = this->mem_tags[i][id];
        if (this->deleted_tags.find(tag) != this->deleted_tags.end()) {
          deleted_ids.insert(id);
        }
      }
      std::cout << "Found " << deleted_ids.size() << " tags to delete from mem-DiskANN #" << i+1 << "\n";
    }
  }

  template<typename T, typename TagT>
  void IndexMerger<T, TagT>::process_deletes() {
    // buf to hold data being read
    char* buf = nullptr;
    alloc_aligned((void**)&buf, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);

    // open output file for writing
    std::string disk_index_name = this->disk_index_out_path + "_disk.index";
    std::cout << "Writing delete consolidated graph to " << disk_index_name << "\n";
    this->output_writer.open(disk_index_name, std::ios::out | std::ios::binary);
    assert(this->output_writer.is_open());
    // skip writing header for now
    this->output_writer.seekp(SECTOR_LEN, std::ios::beg);

    Timer delete_timer;
    // batch consolidate deletes
    std::vector<DiskNode<T>> disk_nodes;
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> id_nhoods;
    uint32_t start_id = 0, new_start_id;
    std::cout << "Consolidating deletes\n";
    
    while(start_id < this->disk_npts) {
      new_start_id = this->disk_index->merge_read(disk_nodes, start_id, SECTORS_PER_MERGE, buf);
 #pragma omp parallel for schedule(dynamic, 128) num_threads(MAX_N_THREADS)
      for(uint32_t i=0; i < disk_nodes.size(); i++) {
        // get thread-specific scratch
        int omp_thread_no = omp_get_thread_num();
        uint8_t* pq_coord_scratch = this->thread_bufs[omp_thread_no];
        assert(pq_coord_scratch != nullptr);
        DiskNode<T> &disk_node = disk_nodes[i];
        this->consolidate_deletes(disk_node, pq_coord_scratch);
      }
      for(auto &disk_node : disk_nodes) {
        if(this->is_deleted(disk_node)) {
          this->free_ids.push_back(disk_node.id);
        }
      }

      uint64_t prev_pos = this->output_writer.tellp();
      this->dump_to_disk(disk_nodes, start_id, buf, SECTORS_PER_MERGE);
      this->output_writer.flush();
      uint64_t cur_pos = this->output_writer.tellp();
      assert(cur_pos - prev_pos == (SECTORS_PER_MERGE * SECTOR_LEN));

      // advance to next block
      disk_nodes.clear();
      id_nhoods.clear();
      std::cout << new_start_id << " / " << this->disk_npts << " nodes processed.\n";
      start_id = new_start_id;
    }
    double e2e_time = ((double) delete_timer.elapsed())/(1000000.0);
    std::cout<<"Processed Deletes in " << e2e_time <<" s." << std::endl;
    std::cout << "Writing header.\n";

    // write header
    this->output_writer.seekp(0, std::ios::beg);
    // HEADER --> [_u64 file size][_u64 nnodes][_u64 medoid ID][_u64 max_node_len][_u64 nnodes_per_sector]
    uint64_t file_size = SECTOR_LEN + (ROUND_UP(ROUND_UP(this->disk_npts, nnodes_per_sector) / nnodes_per_sector, SECTORS_PER_MERGE)) * (uint64_t) SECTOR_LEN;
    *(uint64_t* )(buf) = file_size;
    *(uint64_t* )(buf + sizeof(uint64_t)) = (uint64_t) this->disk_npts;
    // determine medoid
    uint64_t medoid = this->init_ids[0];
    // TODO (correct?, misc) :: better way of selecting new medoid
    while(this->disk_deleted_ids.find(medoid) != this->disk_deleted_ids.end()) {
      std::cout << "Medoid deleted. Choosing another start node.\n";
      auto iter = this->disk_deleted_nhoods.find(medoid);
      assert(iter != this->disk_deleted_nhoods.end());
      medoid = iter->second[0];
    }
    *(uint64_t* )(buf + 2 * sizeof(uint64_t)) = (uint64_t) medoid;
    uint64_t max_node_len = (this->ndims * sizeof(T)) + sizeof(uint32_t) + (this->range * sizeof(uint32_t));
    uint64_t nnodes_per_sector = SECTOR_LEN / max_node_len;
    *(uint64_t* )(buf + 3 * sizeof(uint64_t)) = max_node_len;
    *(uint64_t* )(buf + 4 * sizeof(uint64_t)) = nnodes_per_sector;
    this->output_writer.write(buf, SECTOR_LEN);
    
    // close index
    this->output_writer.close();

    // free buf
    aligned_free((void* ) buf);

    // free backing buf for deletes
    aligned_free((void*)this->delete_backing_buf);
  }

  template<typename T, typename TagT>
  void IndexMerger<T, TagT>::populate_deleted_nhoods() {
    // buf for scratch
    char* buf = nullptr;
    alloc_aligned((void**)&buf, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);

    // scan deleted nodes and get 
    std::vector<DiskNode<T>> deleted_nodes;
    uint64_t backing_buf_size = (uint64_t) this->disk_deleted_ids.size() * ROUND_UP(this->max_node_len, 32);
    backing_buf_size = ROUND_UP(backing_buf_size, 256);
    std::cout << "ALLOC: " << (backing_buf_size << 10) << "KiB aligned buffer for deletes.\n";
    alloc_aligned((void**) &this->delete_backing_buf, backing_buf_size, 256);
    memset(this->delete_backing_buf, 0, backing_buf_size);
    this->disk_index->scan_deleted_nodes(this->disk_deleted_ids, deleted_nodes, buf, this->delete_backing_buf, SECTORS_PER_MERGE);

    // insert into deleted_nhoods
    this->disk_deleted_nhoods.clear();
		this->disk_deleted_nhoods.reserve(deleted_nodes.size());
    for(auto &nhood : deleted_nodes) {
      // WARNING :: ASSUMING DISK GRAPH DEGREE NEVER GOES OVER 512
      assert(nhood.nnbrs < 512);
      std::vector<uint32_t> non_deleted_nbrs;
      for(uint32_t i=0;i<nhood.nnbrs;i++){
        uint32_t id = nhood.nbrs[i];
        auto iter = this->disk_deleted_ids.find(id);
        if (iter == this->disk_deleted_ids.end()) {
          non_deleted_nbrs.push_back(id);
        }
      }
      // this->disk_deleted_nhoods.insert(std::make_pair(nhood.id, std::vector<uint32_t>(nhood.nbrs, nhood.nbrs + nhood.nnbrs)));
      this->disk_deleted_nhoods.insert(std::make_pair(nhood.id, non_deleted_nbrs));
    }

    // free buf
    aligned_free((void*) buf);
    assert(deleted_nodes.size() == this->disk_deleted_ids.size());
    assert(this->disk_deleted_nhoods.size() == this->disk_deleted_ids.size());
  }

  template<typename T, typename TagT>
  void IndexMerger<T, TagT>::consolidate_deletes(DiskNode<T> &disk_node, uint8_t* scratch) {
    // if node is deleted
    if (this->is_deleted(disk_node)) {
      disk_node.nnbrs = 0;
      *(disk_node.nbrs - 1) = 0;
      return;
    }

    const uint32_t id = disk_node.id;
    
    assert(disk_node.nnbrs < 512);
    std::vector<uint32_t> id_nhood(disk_node.nbrs, disk_node.nbrs + disk_node.nnbrs);

    tsl::robin_set<uint32_t> new_edges;

    bool change = false;
    for(auto &nbr : id_nhood) {
      auto iter = this->disk_deleted_nhoods.find(nbr);
      if (iter != this->disk_deleted_nhoods.end()){
        change = true;
        new_edges.insert(iter->second.begin(), iter->second.end());
      } else {
        new_edges.insert(nbr);
      }
    }
    // no refs to deleted nodes --> move to next node
    if (!change){
      return;
    }

    // refs to deleted nodes
    id_nhood.clear();
    id_nhood.reserve(new_edges.size());
    for(auto &nbr : new_edges) {
      // 2nd order deleted edge
      auto iter = this->disk_deleted_ids.find(nbr);
      if (iter != this->disk_deleted_ids.end()) {
        continue;
      } else {
        id_nhood.push_back(nbr);
      }
    }

    // TODO (corner case) :: id_nhood might be empty in adversarial cases
    if (id_nhood.empty()) {
      std::cout << "Adversarial case -- all neighbors of node's neighbors deleted -- ID : " << id << "; exiting\n";
      exit(-1);
    }

    // compute PQ dists and shrink
    std::vector<float> id_nhood_dists(id_nhood.size(), 0.0f);
    assert(scratch != nullptr);
    this->disk_index->compute_pq_dists(id, id_nhood.data(), id_nhood_dists.data(), id_nhood.size(), scratch);

    // prune neighbor list using PQ distances
    std::vector<Neighbor> cand_nbrs(id_nhood.size());
    for(uint32_t i=0;i<id_nhood.size(); i++) {
      cand_nbrs[i].id = id_nhood[i];
      auto iter = this->disk_deleted_ids.find(id_nhood[i]);
      assert(iter == this->disk_deleted_ids.end());
      cand_nbrs[i].distance = id_nhood_dists[i];
    }
    // sort and keep only maxc neighbors
    std::sort(cand_nbrs.begin(), cand_nbrs.end());
    if (cand_nbrs.size() > this->maxc) {
      cand_nbrs.resize(this->maxc);
    }
    std::vector<Neighbor> pruned_nbrs;
    std::vector<float> occlude_factor(cand_nbrs.size(), 0.0f);
    pruned_nbrs.reserve(this->range);
    this->occlude_list_pq(cand_nbrs, id, pruned_nbrs, occlude_factor, scratch);

    // copy back final nbrs
    disk_node.nnbrs = pruned_nbrs.size();
    *(disk_node.nbrs - 1) = disk_node.nnbrs;
    for(uint32_t i = 0; i < pruned_nbrs.size(); i++) {
      disk_node.nbrs[i] = pruned_nbrs[i].id;
      auto iter = this->disk_deleted_ids.find(disk_node.nbrs[i]);
      assert(iter == this->disk_deleted_ids.end());
    }
  }

  template<typename T, typename TagT>
  bool IndexMerger<T, TagT>::is_deleted(const DiskNode<T> &disk_node) {
    // short circuit when disk_node is a `hole` on disk
    if((this->disk_tags[disk_node.id] == std::numeric_limits<uint32_t>::max()) && (disk_node.nnbrs != 0))
	    std::cout << "Node with id " << disk_node.id << " is a hole but has non-zero degree " << disk_node.nnbrs << std::endl;
    return (disk_node.nnbrs == 0) || (this->disk_deleted_ids.find(disk_node.id) != this->disk_deleted_ids.end());
  }
  
  template<typename T, typename TagT>
  void IndexMerger<T, TagT>::compute_rename_map() {
    uint32_t needed = 0;
    for(auto &mem_npt : this->mem_npts) {
      needed += mem_npt;
    }
    for(auto &del_set : this->mem_deleted_ids) {
      needed -= del_set.size();
    }
    std::cout << "RENAME: Need " << needed << ", free: " << this->free_ids.size() << "\n";

    uint32_t last_id = this->disk_npts;
    if (needed > this->free_ids.size()) {
			this->free_ids.reserve(needed);
    }
    while(this->free_ids.size() < needed) {
      this->free_ids.push_back(last_id);
      last_id++;
    }

    // assign free IDs to all new IDs
    std::cout << "RENAME: Assigning IDs.\n";
    uint32_t next_free_index = 0;
		this->rename_map.reserve(needed);
		this->inverse_map.reserve(needed);
    std::vector<std::pair<uint32_t, uint32_t>> rename_pairs(needed);
    std::vector<std::pair<uint32_t, uint32_t>> inverse_pairs(needed);
    for(uint32_t mem_id = 0; mem_id < this->mem_data.size(); mem_id++) {
	   std::cout << "Processing Mem-DiskANN #" << mem_id + 1 << "\n";
      uint32_t offset = this->offset_ids[mem_id];
      const tsl::robin_set<uint32_t> &del_set = this->mem_deleted_ids[mem_id];
      std::vector<bool> deleted(this->mem_npts[mem_id], false);
      for(auto &id : del_set) {
        deleted[id] = true;
      }
      for(uint32_t j=0; j < this->mem_npts[mem_id]; j++){
        // ignore any deleted points
        if (deleted[j]) {
          continue;
        }
        const uint32_t new_id = this->free_ids[next_free_index];
        assert(new_id < last_id);
        rename_pairs[next_free_index].first = offset + j;
        rename_pairs[next_free_index].second = new_id;
        inverse_pairs[next_free_index].first = new_id;
        inverse_pairs[next_free_index].second = offset + j;
        next_free_index++;
      }
    }
    std::cout << "RENAME: Storing mappings for " << next_free_index << " points.\n";
    this->rename_list.clear();
    this->rename_list.reserve(next_free_index);
    this->rename_list.insert(this->rename_list.end(), rename_pairs.begin(), rename_pairs.end());
    this->inverse_list.clear();
    this->inverse_list.reserve(next_free_index);
    this->inverse_list.insert(this->inverse_list.end(), inverse_pairs.begin(), inverse_pairs.end());
    // this->rename_map.clear();
    // this->rename_map.insert(rename_pairs.begin(), rename_pairs.begin() + next_free_index);
    // this->inverse_map.clear();
    // this->inverse_map.insert(inverse_pairs.begin(), inverse_pairs.begin() + next_free_index);
    // TODO (misc) :: fix holes if num deletes > num inserts ??
  }

  template<typename T, typename TagT>
  uint32_t IndexMerger<T, TagT>::rename(uint32_t id) const {
    auto iter = std::lower_bound(this->rename_list.begin(), this->rename_list.end(), 
                                 std::make_pair(id, std::numeric_limits<uint32_t>::max()), 
                                 [] (const auto &left, const auto &right) {return left.first < right.first;});
    if (iter == this->rename_list.end()){
      return std::numeric_limits<uint32_t>::max();
    } else {
      uint32_t idx = std::distance(this->rename_list.begin(), iter);
      const std::pair<uint32_t, uint32_t> &p = this->rename_list[idx];
      if(p.first == id)
        return p.second;
      else
        return std::numeric_limits<uint32_t>::max();
    }
  }
  
  template<typename T, typename TagT>
  uint32_t IndexMerger<T, TagT>::rename_inverse(uint32_t renamed_id) const {
    auto iter = std::lower_bound(this->inverse_list.begin(), this->inverse_list.end(), 
                                 std::make_pair(renamed_id, std::numeric_limits<uint32_t>::max()), 
                                 [] (const auto &left, const auto &right) {return left.first < right.first;});
    if (iter == this->inverse_list.end()){
      return std::numeric_limits<uint32_t>::max();
    } else {
      uint32_t idx = std::distance(this->inverse_list.begin(), iter);
      const std::pair<uint32_t, uint32_t> &p = this->inverse_list[idx];
      if(p.first == renamed_id)
        return p.second;
      else
        return std::numeric_limits<uint32_t>::max();
    }
  }

  template<typename T, typename TagT>
  void IndexMerger<T, TagT>::rename(DiskNode<T> &node) const {
    uint32_t renamed_id = this->rename(node.id);
    if (renamed_id != std::numeric_limits<uint32_t>::max()) {
      node.id = renamed_id;
    }
    uint32_t nnbrs = node.nnbrs;
    for(uint32_t i=0;i<nnbrs;i++) {
      uint32_t renamed_nbr_i = this->rename(node.nbrs[i]);
      if(renamed_nbr_i != std::numeric_limits<uint32_t>::max()) {
        node.nbrs[i] = renamed_nbr_i;
      }
    }
  }

  template<typename T, typename TagT>
  void IndexMerger<T, TagT>::rename(std::vector<uint32_t> &ids) const {
    for(uint32_t i=0;i<ids.size();i++) {
      uint32_t renamed_id = this->rename(ids[i]);
      if (renamed_id != std::numeric_limits<uint32_t>::max()) {
        ids[i] = renamed_id;
      }
    }
  }

  template<typename T, typename TagT>
  const uint32_t IndexMerger<T, TagT>::get_index_id(const uint32_t offset_id) const {
    if (offset_id < this->offset_ids[0]) {
      return std::numeric_limits<uint32_t>::max();
    }
    // should not happen unless some buffer is corrupted
    if (offset_id > this->offset_ids.back() + INDEX_OFFSET) {
      std::cout << "Seen: " << offset_id << ", min: " << offset_ids[0] << ", max: " << offset_ids.back() << "\n";
    }
    assert(offset_id < this->offset_ids.back() + INDEX_OFFSET);
    uint32_t index_no = (offset_id - this->offset_ids[0]) / INDEX_OFFSET;
    assert(index_no < this->offset_ids.size());
    return index_no;
  }

  template<typename T, typename TagT>
  std::vector<uint32_t> IndexMerger<T, TagT>::get_edge_list(const uint32_t offset_id) {
    const uint32_t index_no = this->get_index_id(offset_id);
    if (index_no == std::numeric_limits<uint32_t>::max()) {
      assert(offset_id < this->offset_ids[0]);
      return this->disk_delta->get_nhood(offset_id);
    }
    uint32_t local_id = offset_id - this->offset_ids[index_no];
    assert(local_id < this->mem_npts[index_no]);
    std::vector<uint32_t> ret = this->mem_deltas[index_no]->get_nhood(offset_id);
    // this->rename(ret);
    return ret;
  }

  template<typename T, typename TagT>
  const T* IndexMerger<T, TagT>::get_mem_data(const uint32_t offset_id) {
    const uint32_t index_no = this->get_index_id(offset_id);
    if (index_no == std::numeric_limits<uint32_t>::max()) {
      assert(offset_id < this->offset_ids[0]);
      return nullptr;
    }
    uint32_t local_id = offset_id - this->offset_ids[index_no];
    assert(local_id < this->mem_npts[index_no]);
    return this->mem_data[index_no] + ((uint64_t) local_id * (uint64_t) this->aligned_ndims);
  }

  template<typename T, typename TagT>
  void IndexMerger<T, TagT>::write_tag_file(const std::string &tag_out_filename, const uint32_t npts) {
      diskann::Timer timer;
    std::cout << "Writing new tags to " << tag_out_filename << "\n";
   // std::ofstream tag_writer(tag_out_filename, std::ios::trunc);
    // iterate over num points
    TagT * cur_tags;

    size_t allocSize = npts * sizeof(TagT);
    alloc_aligned(((void **) &cur_tags), allocSize, 8 * sizeof(TagT));

    for(uint32_t i=0; i < npts; i++) {
      TagT cur_tag;
      // check if `i` is in inverse map
      const uint32_t offset_id = this->rename_inverse(i);
      if (offset_id == std::numeric_limits<uint32_t>::max()) {
        cur_tag = this->disk_tags[i];
	if(this->deleted_tags.find(cur_tag) != this->deleted_tags.end())
	{
		*(cur_tags + i) = std::numeric_limits<uint32_t>::max();
	}
	else
		*(cur_tags + i) = cur_tag;
      } else {
        const uint32_t index_no = this->get_index_id(offset_id);
        const uint32_t index_local_id = offset_id - this->offset_ids[index_no];
        cur_tag = this->mem_tags[index_no][index_local_id];
	if(this->deleted_tags.find(cur_tag) != this->deleted_tags.end())
	{
		*(cur_tags + i) = std::numeric_limits<uint32_t>::max();
	}
	else
		*(cur_tags + i) = cur_tag;
      }
//      tag_writer << cur_tag << std::endl;
    }
//    tag_writer.close();
    diskann::save_bin<TagT>(tag_out_filename, cur_tags, npts, 1);
    std::cout << "Tag file written in " << timer.elapsed() << " microsec" << std::endl;
    delete[] cur_tags;
    // release all tags -- automatically deleted since using `unique_ptr`
    this->mem_tags.clear();
  }

  template<typename T, typename TagT>
  void IndexMerger<T, TagT>::process_merges() {
    // buf to hold data being read
    char* buf = nullptr;
    alloc_aligned((void**)&buf, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);

    Timer merge_timer;
    // start at sector=1 in output file for reading + writing
    this->output_writer.seekp(SECTOR_LEN, std::ios::beg);
    uint64_t cur_offset = SECTOR_LEN;
    
    // batch consolidate deletes
    std::vector<DiskNode<T>> disk_nodes;
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> id_nhoods;
    uint32_t start_id = 0, new_start_id;
    std::cout << "Merging inserts into SSD-DiskANN.\n";
    uint64_t delta_avg = 0, delta_max = 0, delta_count = 0;
    std::atomic<uint64_t> counts;
    counts.store(0);
		std::ofstream aux_writer("/tmp/abc");
		// aux_writer << "buf_min = " << (uint64_t) buf << ", buf_max = " << (uint64_t) (buf + SECTORS_PER_MERGE * SECTOR_LEN) << "\n";
    while(start_id < this->disk_npts) {
      // zero buf for better consistency
      memset(buf, 0, SECTORS_PER_MERGE * SECTOR_LEN);
      new_start_id = this->disk_index->merge_read(disk_nodes, start_id, SECTORS_PER_MERGE, buf);
      #pragma omp parallel for schedule(dynamic, 128) num_threads(MAX_N_THREADS)
      for(uint64_t idx = 0; idx < disk_nodes.size(); idx++) {
        // get thread-specific scratch
        int omp_thread_no = omp_get_thread_num();
        uint8_t* thread_scratch = this->thread_bufs[omp_thread_no];

        DiskNode<T> &disk_node = disk_nodes[idx];
        uint32_t id = disk_node.id;

        std::vector<uint32_t> nhood;
				std::vector<uint32_t> deltas;
        uint32_t offset_id = this->rename_inverse(id);
        // replaced by new vector, copy coords and proceed as normal
        if (offset_id != std::numeric_limits<uint32_t>::max()) {
          // copy coords
          const T* vec = this->get_mem_data(offset_id);
          assert(vec != nullptr);
          memcpy(disk_node.coords, vec, this->ndims * sizeof(T));
          disk_node.nnbrs = 0;
          *(disk_node.nbrs - 1) = 0; // also set on buffer
          deltas = this->get_edge_list(offset_id);
          // delta_count++;
          // delta_avg += deltas.size();
          // delta_max = std::max(delta_max, (uint64_t) deltas.size());
        } else {
          // not replaced
          deltas = this->get_edge_list(id);
          delta_count++;
          delta_avg += deltas.size();
          delta_max = std::max(delta_max, (uint64_t) deltas.size());
        }

        // if no edges to add, continue
        if (deltas.empty()) {
          continue;
        }

        uint32_t nnbrs = disk_node.nnbrs;
        nhood.insert(nhood.end(), disk_node.nbrs, disk_node.nbrs + nnbrs);
        nhood.insert(nhood.end(), deltas.begin(), deltas.end());
        // rename nbrs in nhood to use PQ dist comparisons
        // this->rename(nhood); // skipping since get_edge_list() renames delta edges

        // prune neighbor list ONLY if exceeding graph max out-degree (`range`)
        if (nhood.size() > this->range) {
          std::vector<float> dists(nhood.size(), 0.0f);
          std::vector<Neighbor> pool(nhood.size());
          this->disk_index->compute_pq_dists(id, nhood.data(), dists.data(), nhood.size(), thread_scratch);
          for(uint32_t k=0;k<nhood.size();k++) {
            pool[k].id = nhood[k];
            pool[k].distance = dists[k];
						// std::cout << k << ": id=" << nhood[k] << ", dist=" << dists[k] << "\n";
          }
          nhood.clear();
          // prune pool
					std::sort(pool.begin(), pool.end());
          this->prune_neighbors_pq(id, pool, nhood, thread_scratch);
        }
        // copy edges from nhood to disk node
        disk_node.nnbrs = nhood.size();
        // *(disk_node.nbrs - 1) = nhood.size(); // write to buf
        *(disk_node.nbrs - 1) = nhood.size(); // write to buf
				for(uint32_t i=0; i < disk_node.nnbrs; i++) {
					disk_node.nbrs[i] = nhood[i];
				}
        memcpy(disk_node.nbrs, nhood.data(), disk_node.nnbrs * sizeof(uint32_t));
			// 	aux_writer << "disk_node.nbrs =  " << (uint64_t) disk_node.nbrs << ", id = ";
				uint32_t lcounts = 0;
				for(auto &nbr : nhood) {
					if (nbr >= 980000)
						lcounts++;
				}
				/*
				if (lcounts > 0) {
						aux_writer << disk_node.id << ": ";
						for(auto &nbr : nhood) 
										aux_writer << nbr << " ";
						aux_writer << "\n"; 
				}
				*/
				counts += lcounts;
      }
      // dump to disk
      // this->disk_index->passthrough_write(buf, cur_offset, SECTORS_PER_MERGE * SECTOR_LEN);
      cur_offset += SECTORS_PER_MERGE * SECTOR_LEN;
      this->output_writer.write(buf, SECTORS_PER_MERGE * SECTOR_LEN);
      std::cout << new_start_id << " / " << this->disk_npts << " nodes processed.\n";
      start_id = new_start_id;
			aux_writer.flush();
    }
		aux_writer.close();
    std::cout << "Delta statistics:\nMax: " << delta_max << ", Avg: " << (float)delta_avg / (float)delta_count << "\n";
		std::cout << "Old -> new edges: " << counts.load() << "\n";

    // write header
    this->output_writer.seekp(0, std::ios::beg);
    // [_u64 file size][_u64 nnodes][_u64 medoid ID][_u64 max_node_len][_u64 nnodes_per_sector]
    uint64_t file_size = SECTOR_LEN + (ROUND_UP(ROUND_UP(this->disk_npts, nnodes_per_sector) / nnodes_per_sector, SECTORS_PER_MERGE)) * (uint64_t) SECTOR_LEN;
    *(uint64_t* )(buf) = file_size;
    *(uint64_t* )(buf + sizeof(uint64_t)) = (uint64_t) this->disk_npts;
    // determine medoid
    uint64_t medoid = this->init_ids[0];
    /*
    // Don't need medoid correction again
    // TODO (correct?, misc) :: better way of selecting new medoid
    while(this->disk_deleted_ids.find(medoid) != this->disk_deleted_ids.end()) {
      auto iter = this->disk_deleted_nhoods.find(medoid);
      assert(iter != this->disk_deleted_nhoods.end());
      medoid = iter->second[0];
    } */
    *(uint64_t* )(buf + 2 * sizeof(uint64_t)) = (uint64_t) medoid;
    uint64_t max_node_len = this->ndims * sizeof(T) + sizeof(uint32_t) + this->range * sizeof(uint32_t);
    uint64_t nnodes_per_sector = SECTOR_LEN / max_node_len;
    *(uint64_t* )(buf + 3 * sizeof(uint64_t)) = max_node_len;
    *(uint64_t* )(buf + 4 * sizeof(uint64_t)) = nnodes_per_sector;
    this->output_writer.write(buf, SECTOR_LEN);
    
    // close index
    this->output_writer.close();

    // free buf
    aligned_free((void*) buf);
    double e2e_time = ((double) merge_timer.elapsed())/(1000000.0);
    std::cout<<"Time to merge the inserts to disk: "<< e2e_time << "s." << std::endl;
  }

  template<typename T, typename TagT>
  void IndexMerger<T, TagT>::merge() {
    // populate deleted IDs
    this->compute_deleted_ids();

    // BEGIN -- graph on disk has deleted references, maybe some holes
    // populate deleted nodes
    this->populate_deleted_nhoods();
    for(auto &id : this->disk_deleted_ids) {
      auto iter = this->disk_deleted_nhoods.find(id);
      assert(iter != this->disk_deleted_nhoods.end());
    }

    // process all deletes
    this->process_deletes();
    // END -- graph on disk has NO deleted references, maybe some holes

    std::cout << "Computing rename-map.\n";
    // compute rename map
    this->compute_rename_map();

    // get max ID + 1 in rename-map as new max pts
    uint32_t new_max_pts = this->disk_npts - 1;
    /*
    for(auto &k_v : this->rename_map) {
      new_max_pts = std::max(new_max_pts, k_v.second);
    }*/
    // alternative using list
    new_max_pts = std::max(this->inverse_list.back().first, new_max_pts);
    new_max_pts = new_max_pts + 1;

    // TODO (correct) :: figure out naming scheme
    std::string new_disk_out(this->disk_index_out_path + "_disk.index");
    std::cout << "RELOAD: Creating new disk graph at " << new_disk_out << "\n";
    std::string new_pq_prefix(this->disk_index_out_path + "_pq");
    std::string new_pq_coords(new_pq_prefix + "_compressed.bin");
    std::cout << "RELOAD: Creating new PQ coords file " << new_pq_coords << "\n";

    // open same file with 2 different classes
    this->output_writer.close();
		// TODO (correct) :: write to the right file
		std::string tmp_file = TMP_FOLDER + "/index_ravi";

		MallocExtension::instance()->ReleaseFreeMemory();

	    	this->output_writer.open(tmp_file, std::ios::out | std::ios::binary);
    assert(this->output_writer.is_open());

    // BEGIN -- PQ data on disk not consistent, not in right order
    // write outdated PQ data into pq writer with intentionally wrong header
    std::fstream pq_writer(new_pq_coords, std::ios::out | std::ios::binary | std::ios::trunc);
    assert(pq_writer.is_open());
    uint64_t pq_file_size = ((uint64_t) new_max_pts * (uint64_t) this->pq_nchunks) + (2 * sizeof(uint32_t));

    // inflate file size to accommodate new points
    pq_writer.seekp(pq_file_size - sizeof(uint64_t), std::ios::beg);
    pq_writer.write((char*)(&pq_file_size), sizeof(uint64_t));

    // write PQ compressed coords bin and close file
    pq_writer.seekp(0, std::ios::beg);
    uint32_t npts_u32 = new_max_pts, ndims_u32 = this->pq_nchunks;
    pq_writer.write((char* )&npts_u32, sizeof(uint32_t));
    pq_writer.write((char* )&ndims_u32, sizeof(uint32_t));
    pq_writer.write((char* )this->pq_data, (uint64_t) this->disk_npts * (uint64_t) ndims_u32);
    pq_writer.close();

    // write out tags
    const std::string tag_file = new_disk_out + ".tags";
    this->write_tag_file(tag_file, new_max_pts);

    // switch index to read-only mode
    this->disk_index->reload_index(new_disk_out, new_pq_coords, new_max_pts);

    // re-acquire pointers
    auto res = this->disk_index->get_pq_config();
    this->pq_nchunks = res.second;
    this->pq_data = res.first;
    this->disk_npts = this->disk_index->return_nd();
    this->init_ids.clear();
    this->init_ids = this->disk_index->get_init_ids();
    assert(this->disk_npts == new_max_pts);

    // call inserts
    this->process_inserts();
    this->process_inserts_pq();

    std::cout << "Dumping full compressed PQ vectors from memory.\n";
    // re-open PQ writer
    pq_writer.open(new_pq_coords, std::ios::in | std::ios::out | std::ios::binary);
    pq_writer.seekp(2 * sizeof(uint32_t), std::ios::beg);
    // write all (old + new) PQ data to disk; no need to modify header
    pq_writer.write((char* ) this->pq_data, ((uint64_t) new_max_pts * (uint64_t) this->pq_nchunks));
    pq_writer.close();
    // END -- PQ data on disk consistent and in correct order

    // batch rename all inserted edges in each delta
    std::cout << "Renaming edges for easier access during merge.\n";
    // const std::function<uint32_t(uint32_t)> rename_func = std::bind(&IndexMerger<T, TagT>::rename, this);
    const std::function<uint32_t(uint32_t)> rename_func = [this] (uint32_t id) {return this->rename(id);};
    this->disk_delta->rename_edges(rename_func);
    for(auto &delta : this->mem_deltas) {
      delta->rename_edges(rename_func);
    }

    // start merging
    // BEGIN -- graph on disk has NO deleted references, NO newly inserted points
    this->process_merges();
    // END -- graph on disk has NO deleted references, has newly inserted points

		/* copy output from temp_file -> new_disk_out */
		// reset temp_file ptr
		this->output_writer.close();
		// destruct PQFlashIndex
		delete this->disk_index;
		std::cout << "Destroyed PQ Flash Index\n";
    /*
		// open output file
		std::ifstream reader(tmp_file, std::ios::binary);
		std::ofstream final_writer(new_disk_out, std::ios::binary);
		std::cout << "Copying final index: " << tmp_file << " -> " << new_disk_out << "\n";
		// write output
		std::istreambuf_iterator<char> begin_source(reader);
    std::istreambuf_iterator<char> end_source;
    std::ostreambuf_iterator<char> begin_dest(final_writer); 
    // std::copy(begin_source, end_source, begin_dest);
		final_writer << reader.rdbuf();
		// close all files
		final_writer.close();
		reader.close(); */

    auto copy_file = [] (const std::string &src, const std::string &dest) {
      std::cout << "COPY :: " << src << " --> " << dest << "\n";
      std::ofstream dest_writer(dest, std::ios::binary);
      std::ifstream src_reader(src, std::ios::binary);
      dest_writer << src_reader.rdbuf();
      dest_writer.close();
      src_reader.close();
    };
    // copy index
    copy_file(tmp_file, new_disk_out);

    /* copy PQ tables */
    std::string prefix_pq_in = this->disk_index_in_path + "_pq";
    std::string prefix_pq_out = this->disk_index_out_path + "_pq";
    // PQ pivots
    copy_file(prefix_pq_in + "_pivots.bin", prefix_pq_out + "_pivots.bin");
    // PQ pivots centroid
    copy_file(prefix_pq_in + "_pivots.bin_centroid.bin", prefix_pq_out + "_pivots.bin_centroid.bin");
    // PQ pivots chunk offsets
    copy_file(prefix_pq_in + "_pivots.bin_chunk_offsets.bin", prefix_pq_out + "_pivots.bin_chunk_offsets.bin");
    // PQ pivots re-arrangement permutation
    copy_file(prefix_pq_in + "_pivots.bin_rearrangement_perm.bin", prefix_pq_out + "_pivots.bin_rearrangement_perm.bin");
  }

  // template class instantiations
  template class IndexMerger<float, uint32_t>;
  template class IndexMerger<uint8_t, uint32_t>;
  template class IndexMerger<int8_t, uint32_t>;
  template class IndexMerger<float, int64_t>;
  template class IndexMerger<uint8_t, int64_t>;
  template class IndexMerger<int8_t, int64_t>;
  template class IndexMerger<float, uint64_t>;
  template class IndexMerger<uint8_t, uint64_t>;
  template class IndexMerger<int8_t, uint64_t>;
} // namespace diskann
