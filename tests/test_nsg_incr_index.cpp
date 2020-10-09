#include <index.h>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <sync_index.h>
#include <time.h>
#include <timer.h>

#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "memory_mapper.h"

template<typename T>
int build_incremental_index(const std::string& data_path, const unsigned L,
                            const unsigned R, const unsigned C,
                            const unsigned num_rnds, const float alpha,
                            const std::string& save_path,
                            const unsigned     num_incr,
                            const unsigned     num_frozen) {
  diskann::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("saturate_graph", false);
  paras.Set<unsigned>("num_rnds", num_rnds);

  T*     data_load = NULL;
  size_t num_points, dim, aligned_dim;

  diskann::load_aligned_bin<T>(data_path.c_str(), data_load, num_points, dim,
                               aligned_dim);

  typedef int TagT;

  diskann::Index<T, TagT> index(diskann::L2, dim, num_points, num_frozen, true,
                                true, false);
  {
    std::vector<TagT> tags(num_points - num_incr);
    std::iota(tags.begin(), tags.end(), 0);

    diskann::Timer timer;
    index.build(data_path.c_str(), num_points - num_incr, paras, tags);
    std::cout << "Index build time: " << timer.elapsed() / 1000 << "ms\n";
    index.save(save_path.c_str());
  }

  {
    diskann::Timer timer;
#pragma omp        parallel for
    for (size_t i = num_points - num_incr; i < num_points; ++i) {
      index.insert_point(data_load + i * aligned_dim, paras, i);
    }
    std::cout << "Incremental time: " << timer.elapsed() / 1000 << "ms\n";
    auto save_path_inc = save_path + ".inc";
    index.save(save_path_inc.c_str());
  }

  tsl::robin_set<unsigned> delete_list;
  while (delete_list.size() < num_incr)
    delete_list.insert(rand() % num_points);
  std::cout << "Deleting " << delete_list.size() << " elements" << std::endl;
  std::vector<unsigned> delete_vector;

  for (auto p : delete_list) {
    delete_vector.emplace_back(p);
  }
  std::cout << "Size of delete_vector : " << delete_vector.size() << std::endl;
  {
    index.enable_delete();
    //    diskann::Timer timer;
    //#pragma omp        parallel for
    for (size_t i = 0; i < delete_vector.size(); i++) {
      unsigned p = delete_vector[i];
      //      if (index.eager_delete(p, paras) != 0)
      if (index.lazy_delete(p) != 0)
        std::cerr << "Delete tag " << p << " not found" << std::endl;
    }
    //    std::cout << "Delete time : " << timer.elapsed() / 1000 << " ms\n";

    diskann::Timer timer;
    if (index.disable_delete(paras, true) != 0) {
      std::cerr << "Disable delete failed" << std::endl;
      return -1;
    }

    std::cout << "Delete time : " << timer.elapsed() / 1000 << " ms\n";
  }

  auto save_path_del = save_path + ".del";
  index.save(save_path_del.c_str());

  {
    diskann::Timer timer;
    for (auto p : delete_list) {
      index.insert_point(data_load + (size_t) p * (size_t) aligned_dim, paras,
                         p);
    }
    std::cout << "Re-incremental time: " << timer.elapsed() / 1000 << "ms\n";
  }

  auto save_path_reinc = save_path + ".reinc";
  index.save(save_path_reinc.c_str());

  delete[] data_load;

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 11) {
    std::cout << "Correct usage: " << argv[0]
              << " type[int8/uint8/float] data_file L R C alpha "
                 "num_rounds "
              << "save_graph_file #incr_points #frozen_points " << std::endl;
    exit(-1);
  }

  unsigned    L = (unsigned) atoi(argv[3]);
  unsigned    R = (unsigned) atoi(argv[4]);
  unsigned    C = (unsigned) atoi(argv[5]);
  float       alpha = (float) std::atof(argv[6]);
  unsigned    num_rnds = (unsigned) std::atoi(argv[7]);
  std::string save_path(argv[8]);
  unsigned    num_incr = (unsigned) atoi(argv[9]);
  unsigned    num_frozen = (unsigned) atoi(argv[10]);

  if (std::string(argv[1]) == std::string("int8"))
    build_incremental_index<int8_t>(argv[2], L, R, C, num_rnds, alpha,
                                    save_path, num_incr, num_frozen);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_incremental_index<uint8_t>(argv[2], L, R, C, num_rnds, alpha,
                                     save_path, num_incr, num_frozen);
  else if (std::string(argv[1]) == std::string("float"))
    build_incremental_index<float>(argv[2], L, R, C, num_rnds, alpha, save_path,
                                   num_incr, num_frozen);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
