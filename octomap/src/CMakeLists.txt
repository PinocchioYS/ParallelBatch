SET (octomap_SRCS
  AbstractOcTree.cpp
  AbstractOccupancyOcTree.cpp
  Pointcloud.cpp
  ScanGraph.cpp
  CountingOcTree.cpp
  OcTree.cpp
  OcTreeNode.cpp
  OcTreeStamped.cpp
  ColorOcTree.cpp
  )

# dynamic and static libs, see CMake FAQ:
ADD_LIBRARY( octomap SHARED ${octomap_SRCS})
set_target_properties( octomap PROPERTIES
  VERSION ${OCTOMAP_VERSION}
  SOVERSION ${OCTOMAP_SOVERSION}
)
ADD_LIBRARY( octomap-static STATIC ${octomap_SRCS})
SET_TARGET_PROPERTIES(octomap-static PROPERTIES OUTPUT_NAME "octomap") 

TARGET_LINK_LIBRARIES(octomap octomath)

if(NOT EXISTS "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cmake/octomap")
  file(MAKE_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cmake/octomap")
endif()

export(TARGETS octomap octomap-static
  APPEND FILE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cmake/octomap/octomap-targets.cmake")

# ADD_EXECUTABLE(example_parallelbatchOcTree example_parallelbatchOcTree.cpp)
# TARGET_LINK_LIBRARIES(example_parallelbatchOcTree octomap)

install(TARGETS octomap octomap-static
  EXPORT octomap-targets
  INCLUDES DESTINATION include
  ${INSTALL_TARGETS_DEFAULT_ARGS}
)
install(EXPORT octomap-targets DESTINATION share/octomap/)

install(TARGETS
	graph2tree
	log2graph
	binvox2bt
	bt2vrml
	edit_octree
	convert_octree
	eval_octree_accuracy
	compare_octrees
	${INSTALL_TARGETS_DEFAULT_ARGS}
)

