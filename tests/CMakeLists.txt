set(CMAKE_CXX_STANDARD 14)

#add_executable(test_merge_insert test_merge_insert.cpp)
#if(MSVC)

#	target_link_options(test_merge_insert PRIVATE /MACHINE:x64)
#	target_link_libraries(test_merge_insert debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib)
#	target_link_libraries(test_merge_insert optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib)
#else() 
#	target_link_libraries(test_merge_insert ${PROJECT_NAME} -ltcmalloc aio)
#endif()

add_executable(test_concurr_merge_insert test_concurr_merge_insert.cpp)
if(MSVC)

	target_link_options(test_concurr_merge_insert PRIVATE /MACHINE:x64)
	target_link_libraries(test_concurr_merge_insert debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib)
	target_link_libraries(test_concurr_merge_insert optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib)
else() 
	target_link_libraries(test_concurr_merge_insert ${PROJECT_NAME} -ltcmalloc aio)
endif()

add_executable(merge_in_mem merge_in_mem.cpp)
if(MSVC)

	target_link_options(merge_in_mem PRIVATE /MACHINE:x64)
	target_link_libraries(merge_in_mem debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib)
	target_link_libraries(merge_in_mem optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib)
else() 
    target_link_libraries(merge_in_mem ${PROJECT_NAME} -ltcmalloc aio)
endif()

add_executable(test_nsg_incr_index test_nsg_incr_index.cpp)
if(MSVC)

	target_link_options(test_nsg_incr_index PRIVATE /MACHINE:x64)
	target_link_libraries(test_nsg_incr_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib)
	target_link_libraries(test_nsg_incr_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib)
else() 
    target_link_libraries(test_nsg_incr_index ${PROJECT_NAME} -ltcmalloc aio)
endif()

add_executable(lazy_reinsert_batch lazy_reinsert_batch.cpp)
if(MSVC)

	target_link_options(lazy_reinsert_batch PRIVATE /MACHINE:x64)
	target_link_libraries(lazy_reinsert_batch debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib)
	target_link_libraries(lazy_reinsert_batch optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib)
else() 
    target_link_libraries(lazy_reinsert_batch ${PROJECT_NAME} -ltcmalloc aio)
endif()

add_executable(static_sync_index static_sync_index.cpp)
if(MSVC)

	target_link_options(static_sync_index PRIVATE /MACHINE:x64)
	target_link_libraries(static_sync_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib)
	target_link_libraries(static_sync_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib)
else() 
	target_link_libraries(static_sync_index ${PROJECT_NAME} -ltcmalloc aio)
endif()

add_executable(test_index_merger test_index_merger.cpp)
if(MSVC)

	target_link_options(test_index_merger PRIVATE /MACHINE:x64)
	target_link_libraries(test_index_merger debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib)
	target_link_libraries(test_index_merger optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib)
else() 
	target_link_libraries(test_index_merger ${PROJECT_NAME} -ltcmalloc aio)
endif()

add_executable(test_index_merger_steady_state test_index_merger_steady_state.cpp)
if(MSVC)

	target_link_options(test_index_merger_steady_state PRIVATE /MACHINE:x64)
	target_link_libraries(test_index_merger_steady_state debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib)
	target_link_libraries(test_index_merger_steady_state optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib)
else() 
	target_link_libraries(test_index_merger_steady_state ${PROJECT_NAME} -ltcmalloc aio)
endif()


add_executable(test_sync_index_batch test_sync_index_batch.cpp)
if(MSVC)

	target_link_options(test_sync_index_batch PRIVATE /MACHINE:x64)
	target_link_libraries(test_sync_index_batch debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib)
	target_link_libraries(test_sync_index_batch optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib)
else() 
	target_link_libraries(test_sync_index_batch ${PROJECT_NAME} -ltcmalloc aio)
endif()

add_executable(test_sync_index test_sync_index.cpp)
if(MSVC)

	target_link_options(test_sync_index PRIVATE /MACHINE:x64)
	target_link_libraries(test_sync_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib)
	target_link_libraries(test_sync_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib)
else() 
	target_link_libraries(test_sync_index ${PROJECT_NAME} -ltcmalloc aio)
endif()

add_executable(build_memory_index build_memory_index.cpp)
if(MSVC)
	target_link_options(build_memory_index PRIVATE /MACHINE:x64 /DEBUG:FULL "/INCLUDE:_tcmalloc")
	target_link_libraries(build_memory_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
	target_link_libraries(build_memory_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
else() 
	target_link_libraries(build_memory_index ${PROJECT_NAME} -ltcmalloc)
endif()

add_executable(search_memory_index search_memory_index.cpp)
if(MSVC)
	target_link_options(search_memory_index PRIVATE /MACHINE:x64 /DEBUG:FULL)
	target_link_libraries(search_memory_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib)
	target_link_libraries(search_memory_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib)
else() 
	target_link_libraries(search_memory_index ${PROJECT_NAME} aio -ltcmalloc)
endif()

add_executable(build_disk_index build_disk_index.cpp)
if(MSVC)
	target_link_options(build_disk_index PRIVATE /MACHINE:x64 /DEBUG:FULL "/INCLUDE:_tcmalloc")
	target_link_libraries(build_disk_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
	target_link_libraries(build_disk_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
else() 
	target_link_libraries(build_disk_index ${PROJECT_NAME} -ltcmalloc aio)
endif()

add_executable(search_disk_index search_disk_index.cpp
    ${PROJECT_SOURCE_DIR}/src/aux_utils.cpp)
if(MSVC)
	target_link_options(search_disk_index PRIVATE /MACHINE:x64 /DEBUG:FULL)
	target_link_libraries(search_disk_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib)
	target_link_libraries(search_disk_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib)
else() 
	target_link_libraries(search_disk_index ${PROJECT_NAME} aio -ltcmalloc)
endif()


#if(MSVC)
#	add_executable(test_diskpriority_io test_diskpriority_io.cpp)
#	target_include_directories(test_diskpriority_io PRIVATE "F:/boost/boost_1_71_0")
#	target_link_options(test_diskpriority_io PRIVATE /MACHINE:x64 /DEBUG:FULL)
#	target_link_libraries(test_diskpriority_io debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskpriority_io.lib)
#	target_link_libraries(test_diskpriority_io optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskpriority_io.lib)
#endif()

# formatter
