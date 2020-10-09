#pragma once
#include "utils.h"

namespace diskann {
  template<typename T>
  class FixedChunkPQTable {
    // data_dim = n_chunks * chunk_size;
    float* tables =
        nullptr;  // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    //    _u64   n_chunks;    // n_chunks = # of chunks ndims is split into
    //    _u64   chunk_size;  // chunk_size = chunk size of each dimension chunk
    _u64   ndims;  // ndims = chunk_size * n_chunks
    _u64   n_chunks;
    _u32*  chunk_offsets = nullptr;
    _u32*  rearrangement = nullptr;
    float* centroid = nullptr;
    float* tables_T = nullptr;  // same as pq_tables, but col-major
    float* all_to_all_dists = nullptr;

   public:
    FixedChunkPQTable() {
    }

    virtual ~FixedChunkPQTable() {
      if (tables != nullptr)
        delete[] tables;
      if (tables_T != nullptr)
        delete[] tables_T;
      if (rearrangement != nullptr)
        delete[] rearrangement;
      if (chunk_offsets != nullptr)
        delete[] chunk_offsets;
      if (centroid != nullptr)
        delete[] centroid;
      if (all_to_all_dists != nullptr)
        delete[] all_to_all_dists;
    }

    void load_pq_centroid_bin(const char* pq_table_file, size_t num_chunks) {
      std::string rearrangement_file =
          std::string(pq_table_file) + "_rearrangement_perm.bin";
      std::string chunk_offset_file =
          std::string(pq_table_file) + "_chunk_offsets.bin";
      std::string centroid_file = std::string(pq_table_file) + "_centroid.bin";

      // bin structure: [256][ndims][ndims(float)]
      uint64_t numr, numc;
      size_t   npts_u64, ndims_u64;
      diskann::load_bin<float>(pq_table_file, tables, npts_u64, ndims_u64);
      this->ndims = ndims_u64;

      if (file_exists(chunk_offset_file)) {
        diskann::load_bin<_u32>(rearrangement_file, rearrangement, numr, numc);
        if (numr != ndims_u64 || numc != 1) {
          std::cout << "Error loading rearrangement file" << std::endl;
          throw diskann::ANNException("Error loading rearrangement file", -1,
                                      __FUNCSIG__, __FILE__, __LINE__);
        }

        diskann::load_bin<_u32>(chunk_offset_file, chunk_offsets, numr, numc);
        if (numc != 1 || numr != num_chunks + 1) {
          std::cout << "Error loading chunk offsets file" << std::endl;
          throw diskann::ANNException("Error loading chunk offsets file", -1,
                                      __FUNCSIG__, __FILE__, __LINE__);
        }

        this->n_chunks = numr - 1;

        diskann::load_bin<float>(centroid_file, centroid, numr, numc);
        if (numc != 1 || numr != ndims_u64) {
          std::cout << "Error loading centroid file" << std::endl;
          throw diskann::ANNException("Error loading centroid file", -1,
                                      __FUNCSIG__, __FILE__, __LINE__);
        }
      } else {
        this->n_chunks = num_chunks;
        rearrangement = new uint32_t[ndims];

        uint64_t chunk_size = DIV_ROUND_UP(ndims, num_chunks);
        for (uint32_t d = 0; d < ndims; d++)
          rearrangement[d] = d;
        chunk_offsets = new uint32_t[num_chunks + 1];
        for (uint32_t d = 0; d <= num_chunks; d++)
          chunk_offsets[d] = (_u32)(std::min)(ndims, d * chunk_size);
        centroid = new float[ndims];
        std::memset(centroid, 0, ndims * sizeof(float));
      }

      //      std::cout << "PQ Pivots: #ctrs: " << npts_u64 << ", #dims: " <<
      //      ndims_u64
      //                << ", #chunks: " << n_chunks << std::endl;
      //      assert((_u64) ndims_u32 == n_chunks * chunk_size);
      // alloc and compute transpose
      tables_T = new float[256 * ndims_u64];
      for (_u64 i = 0; i < 256; i++) {
        for (_u64 j = 0; j < ndims_u64; j++) {
          tables_T[j * 256 + i] = tables[i * ndims_u64 + j];
        }
      }

      // ravi: added this for easy PQ-PQ squared-distance calculations
      all_to_all_dists = new float[256 * 256 * n_chunks];
      std::memset(all_to_all_dists, 0, 256 * 256 * n_chunks * sizeof(float));
      // should perhaps optimize later
      for (_u32 i = 0; i < 256; i++) {
        for (_u32 j = 0; j < 256; j++) {
          for (_u32 c = 0; c < n_chunks; c++) {
            for (_u64 d = chunk_offsets[c]; d < chunk_offsets[c + 1]; d++) {
              float diff =
                  (tables[i * ndims_u64 + d] - tables[j * ndims_u64 + d]);
              all_to_all_dists[i * 256 * n_chunks + j * n_chunks + c] +=
                  diff * diff;
            }
          }
        }
      }
    }

    void populate_chunk_distances(const T* query_vec, float* dist_vec) {
      memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
      // chunk wise distance computation
      for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
        // sum (q-c)^2 for the dimensions associated with this chunk
        float* chunk_dists = dist_vec + (256 * chunk);
        for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
          _u64         permuted_dim_in_query = rearrangement[j];
          const float* centers_dim_vec = tables_T + (256 * j);
          for (_u64 idx = 0; idx < 256; idx++) {
            float diff = centers_dim_vec[idx] -
                         ((float) query_vec[permuted_dim_in_query] -
                          centroid[permuted_dim_in_query]);
            chunk_dists[idx] += (diff * diff);
          }
        }
      }
    }

    // computes PQ distance between comp_src and comp_dsts in efficient manner
    // comp_src: [nchunks]
    // comp_dsts: count * [nchunks]
    // dists: [count]
    // TODO (perf) :: re-order computation to get better locality
    void compute_distances(const _u8* comp_src, const _u8* comp_dsts,
                           float* dists, const _u32 count) {
      std::memset(dists, 0, count * sizeof(float));
      for (_u64 i = 0; i < count; i++) {
        for (_u64 c = 0; c < n_chunks; c++) {
          dists[i] +=
              all_to_all_dists[(_u64) comp_src[c] * 256 * n_chunks +
                               (_u64) comp_dsts[i * n_chunks + c] * n_chunks +
                               c];
        }
      }
    }

    // fp_vec: [ndims]
    // out_pq_vec : [nchunks]
    void deflate_vec(const float* fp_vec, _u8* out_pq_vec) {
      // permute the vector according to PQ rearrangement, compute all distances
      // to 256 centroids and choose the closest (for each chunk)
      for (_u32 c = 0; c < n_chunks; c++) {
        float closest_dist = std::numeric_limits<float>::max();
        for (_u32 i = 0; i < 256; i++) {
          float cur_dist = 0;
          for (_u64 d = chunk_offsets[c]; d < chunk_offsets[c + 1]; d++) {
            float diff =
                (tables[i * ndims + d] - ((float) fp_vec[rearrangement[d]] -
                                          centroid[rearrangement[d]]));
            cur_dist += diff * diff;
          }
          if (cur_dist < closest_dist) {
            closest_dist = cur_dist;
            out_pq_vec[c] = i;
          }
        }
      }
    }
  };
}  // namespace diskann
