set(CMAKE_CXX_STANDARD 14)

if(MSVC)
	add_subdirectory(dll)
	add_subdirectory(webservice)
else()
	#file(GLOB CPP_SOURCES *.cpp)
	set(CPP_SOURCES ann_exception.cpp aux_utils.cpp index.cpp
        linux_aligned_file_reader.cpp math_utils.cpp memory_mapper.cpp
        partition_and_pq.cpp pq_flash_index.cpp shard.cpp sync_index.cpp
        v2/graph_delta.cpp v2/index_merger.cpp v2/mem_aligned_file_reader.cpp
        v2/merge_insert.cpp)
	add_library(${PROJECT_NAME} ${CPP_SOURCES})
	add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
endif()
install()
