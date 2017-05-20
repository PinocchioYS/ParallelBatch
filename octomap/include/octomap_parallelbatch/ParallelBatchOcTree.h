/*
* Copyright(c) 2017, Youngsun Kwon, Donghyuk Kim, and Sung-eui Yoon, KAIST
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met :
*
*     * Redistributions of source code must retain the above copyright notice, this
*       list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright notice,
*       this list of conditions and the following disclaimer in the documentation
*       and / or other materials provided with the distribution.
*     * Neither the name of ParallelBatch nor the names of its
*       contributors may be used to endorse or promote products derived from
*       this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*/

#ifndef OCTOMAP_PARALLELBATCH_OCTREE_H
#define OCTOMAP_PARALLELBATCH_OCTREE_H

#include <octomap/octomap.h>
// #include <octomap_parallelbatch/RayDistributor.h>
#include <queue>

namespace octomap{
	class ParallelBatchOcTree : public OccupancyOcTreeBase<OcTreeNode> {
	public:
		/// Default constructor, sets resolution of leafs
		ParallelBatchOcTree(double resolution);

		/**
		* Reads an OcTree from a binary file
		* @param _filename
		*
		*/
		ParallelBatchOcTree(std::string _filename);

		virtual ~ParallelBatchOcTree(){};

		/// virtual constructor: creates a new object of same type
		/// (Covariant return type requires an up-to-date compiler)
		ParallelBatchOcTree* create() const { return new ParallelBatchOcTree(resolution); }

		std::string getTreeType() const { return "ParallelBatchOcTree"; }

		// Ray distribution to prallel batching-based updates

		/**
		* Integrate a Pointcloud (in global reference frame), parallelized with OpenMP.
		* Special care is taken that each voxel
		* in the map is updated only once, and occupied nodes have a preference over free ones.
		* This function simply inserts all rays of the point clouds in batch manner, similar to insertPointCloud of octomap::OcTree.
		* Occupied nodes have a preference over free ones.
		*
		* @param scan Pointcloud (measurement endpoints), in global reference frame
		* @param sensor_origin measurement origin in global reference frame
		* @param NUM_OF_THREADS number of threads to be used for parallelization
		*/
		virtual void insertPointCloud(const Pointcloud& scan, const point3d& origin, const int NUM_OF_THREADS);

		/**
		* Integrate a Pointcloud (in global reference frame) using ray distribution, parallelized with OpenMP.
		* This function distributes a point clouds into threads for exploiting the parallelized batching.
		* Special care is taken that each voxel
		* in the map is updated only once, and occupied nodes have a preference over free ones.
		*
		* @param scan Pointcloud (measurement endpoints), in global reference frame
		* @param sensor_origin measurement origin in global reference frame
		* @param NUM_OF_THREADS number of threads to be used for parallelization
		*/
		virtual void insertPointCloudParallelBatch(const Pointcloud& scan, const point3d& origin, const int NUM_OF_THREADS);

	protected:
		// A point in spherical coordinate with unit radius, associated with a ray of point clouds
		struct SphPoint{
			// Constructor
			SphPoint(void) : point(NULL), pi(0.0), theta(0.0), workload(0) {}
			SphPoint(point3d_collection* _p, const double& _pi, const double& _theta, const int& _workload) : point(_p), pi(_pi), theta(_theta), workload(_workload) {}
			SphPoint(const SphPoint& _o) : point(_o.point), pi(_o.pi), theta(_o.theta), workload(_o.workload) {}

			// Reference to a ray of point clouds in world coordinate
			point3d_collection* point;

			// Spherical coordinate with unit radius
			double pi;
			double theta;
			// Workload
			int workload;
		};
		typedef std::vector<SphPoint> SphPointcloud;
		typedef std::vector<SphPoint*> SphRefVector;

		// Comparators for SphRefVector(=std::vector<SphPoint*>)
		class ComparePI : std::unary_function<SphPoint*, bool> {
		public:
			bool operator()(const SphPoint* lhs, const SphPoint* rhs) const {
				return lhs->pi < rhs->pi;
			}
		};
		class CompareTHETA : std::unary_function<SphPoint*, bool> {
		public:
			bool operator()(const SphPoint* lhs, const SphPoint* rhs) const {
				return lhs->theta < rhs->theta;
			}
		};

		// Task that thread should process the ray tracing and batching
		struct Task{
			// Constructor
			Task(const SphRefVector::iterator& _s, const SphRefVector::iterator& _e, const int& _wl) : startIt(_s), endIt(_e), totalWorkload(_wl) {}

			// Information of work to be processed
			SphRefVector::iterator startIt;
			SphRefVector::iterator endIt;
			int totalWorkload;
		};
		typedef std::vector<Task*> TaskSet;

		// Data structure for batching
		typedef unordered_ns::unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash> BatchList;

		/**
		* Convert rays in sensor coordinate to points in spherical coordinate with unit radius
		* In order to distribute the rays fast, we use voxelized point clouds.
		*
		* @param _origin measurement origin in global reference frame
		* @param _voxPC voxelized point clouds in global reference frame
		* @param _sphPC point clouds in spherical coordinate with unit radius
		* @return maximum amount of workloads of input point clouds
		*/
		typedef unordered_ns::unordered_map<octomap::OcTreeKey, point3d_collection, octomap::OcTreeKey::KeyHash> VoxPointclouds;
		int Sensor2Spherical(const point3d& _origin, const VoxPointclouds& _voxPC, SphPointcloud& _sphPC);

		/**
		* Find an axis used for distributing points in spherical coordinate.
		*
		* @param _task points to be distributed
		* @return index of axis (0: pi, 1: theta)
		*/
		int findSplitAxis(const Task& _task);

		/**
		* Find a split point on the axis, according to criteria from workload balancing
		*
		* @param _task points to be distributed
		* @return index of split point
		*/
		SphRefVector::iterator findSplitPoint(const Task& _task);

		/**
		* Static member object which ensures that this OcTree's prototype
		* ends up in the classIDMapping only once. You need this as a
		* static member in any derived octree class in order to read .ot
		* files through the AbstractOcTree factory. You should also call
		* ensureLinking() once from the constructor.
		*/
		class StaticMemberInitializer{
		public:
			StaticMemberInitializer() {
				ParallelBatchOcTree* tree = new ParallelBatchOcTree(0.1);
				tree->clearKeyRays();
				AbstractOcTree::registerTreeType(tree);
			}

			/**
			* Dummy function to ensure that MSVC does not drop the
			* StaticMemberInitializer, causing this tree failing to register.
			* Needs to be called from the constructor of this octree.
			*/
			void ensureLinking() {};
		};

		/// to ensure static initialization (only once)
		static StaticMemberInitializer parallelbatchOcTreeMemberInit;
	};
}

#endif
