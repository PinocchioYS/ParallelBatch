ParallelBatch - High performance method for updating tree-based occupancy maps using multi-threads
================================================================================

http://sglab.kaist.ac.kr/projects/ParallelBatch

ParallelBatch is a library for efficiently updating tree-based occupancy map representation using multi-threads.
The update approach using parallelized batching shows high performance without compromising representation accuracy.
The implementation of our library is based on [OctoMap library](https://github.com/Octomap/octomap).
You can see the detailed information of batching based method in parallel manner at [here](http://sglab.kaist.ac.kr/ParallelBatch).

License
-------
* ParallelBatch: [NEW BSD License](LICENSE.txt)
* OctoMap: [New BSD License](octomap/LICENSE.txt)

BUILD
-----
You can build the ParallelBatch and OctoMap library together with CMake in the top-level directory.
The implementation of recent ParallelBatch 1.0.0 depends on OctoMap 1.9.0.
E.g. for compiling the library, run:

	cd ParallelBatch-master
	mkdir build
	cd build
	cmake ..
	make

Binaries and libs will end up in the directories 'bin' and 'lib' of the top-level directory where you started the build.

See [octomap README](octomap/README.md) for further details and hints on compiling.
Authors of OctoMap library describe how to compile the libraries on various platforms and IDEs.

We tested compiling on MSVC2013.
If you have any problem or issue, notice it at [here](https://github.com/PinocchioYS/ParallelBatch/issues).