#include "mem_aligned_file_reader.h"
#include <cassert>
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */

namespace diskann {
  MemAlignedFileReader::MemAlignedFileReader() {
//	  std::cout << "constr\n";
  }
  MemAlignedFileReader::~MemAlignedFileReader() {
  }

  IOContext& MemAlignedFileReader::get_ctx() {
    return this->bad_ctx;
  }

  // register thread-id for a context
  void MemAlignedFileReader::register_thread() {
//	  std::cout << "registering thread \n";
  }

  // de-register thread-id for a context
  void MemAlignedFileReader::deregister_thread() {

  }

  // Open & close ops
  // Blocking calls
  void MemAlignedFileReader::open(const std::string &fname, bool enable_writes = false, bool enable_create = false) {
    int flags = O_LARGEFILE;
    if (!enable_writes){
      flags |= O_RDONLY;
    } else {
      flags |= O_RDWR;
    }
    if (enable_create) {
      flags |= O_CREAT;
    }
    this->file_desc = ::open(fname.c_str(), flags);
    if (this->file_desc == -1) {
			std::cout << "failed to open mem-file: errno=" << errno << "\n";
    }
    // error checks
    assert(this->file_desc != -1);
    std::cerr << "Opened mem-file : " << fname << std::endl;
  }

  void MemAlignedFileReader::close() {
    // check to make sure file_desc is closed
    int ret = ::fcntl(this->file_desc, F_GETFD);
    assert(ret != -1);

    ret = ::close(this->file_desc);
    assert(ret != -1);
  }

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  void MemAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                  IOContext &ctx, bool async = false) {
    for(auto &read_req : read_reqs) {
      char* buf = (char*) read_req.buf;
      uint64_t offset = read_req.offset;
      uint64_t len = read_req.len;
      int ret = pread64(this->file_desc, buf, len, offset);
      assert(ret != -1);
    }
  }

  void MemAlignedFileReader::sequential_write(AlignedRead &write_req, IOContext& ctx) {
    uint64_t offset = write_req.offset;
    uint64_t len = write_req.len;
    char* buf = (char* ) write_req.buf;
    int ret = pwrite64(this->file_desc, buf, len, offset);
    assert(ret != -1);
  }
} // namespace diskann
