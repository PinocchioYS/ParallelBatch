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

#include <octomap_parallelbatch/ParallelBatchOcTree.h>
#include <cfloat>

namespace octomap{
    ParallelBatchOcTree::ParallelBatchOcTree(double resolution)
            : OccupancyOcTreeBase<OcTreeNode>(resolution) {
        parallelbatchOcTreeMemberInit.ensureLinking();
    };

    ParallelBatchOcTree::ParallelBatchOcTree(std::string _filename)
            : OccupancyOcTreeBase<OcTreeNode>(0.1)  { // resolution will be set according to tree file
        readBinary(_filename);
    }

    ParallelBatchOcTree::StaticMemberInitializer ParallelBatchOcTree::parallelbatchOcTreeMemberInit;

    void ParallelBatchOcTree::insertPointCloud(const Pointcloud& pc, const point3d& origin, const int NUM_OF_THREADS)
    {
        if (pc.size() < 1 || NUM_OF_THREADS < 1)
            return;

        omp_set_num_threads(NUM_OF_THREADS);

        std::vector<octomap::KeyRay> keyrays(NUM_OF_THREADS);
        BatchList free_cells, occupied_cells;

#pragma omp parallel for schedule(guided)
        for (int i = 0; i < (int)pc.size(); ++i){
            const octomap::point3d& p = pc[i];

            unsigned int threadIdx = omp_get_thread_num();

            KeyRay* keyray = &(this->keyrays.at(threadIdx));

            if (this->computeRayKeys(origin, p, *keyray)){
#pragma omp critical(free_insert)
                {
                    // Batch free cells
                    for (octomap::KeyRay::iterator it = keyray->begin(); it != keyray->end(); it++){
                        const BatchList::iterator& cell = free_cells.find(*it);
                        if (cell == free_cells.end())
                            free_cells.insert(std::pair<octomap::OcTreeKey, int>(*it, 1));
                        else
                            cell->second = cell->second + 1;
                    }
                }

#pragma omp critical(occupied_insert)
                {
                    // Batch occupied cells
                    octomap::OcTreeKey key = coordToKey(p);
                    const BatchList::iterator& cell = occupied_cells.find(key);
                    if (cell == occupied_cells.end())
                        occupied_cells.insert(std::pair<octomap::OcTreeKey, int>(key, 1));
                    else
                        cell->second = cell->second + 1;
                }
            }
        }

        // Update free cells first, and then update occupied cells
        for (BatchList::iterator it = free_cells.begin(); it != free_cells.end(); ++it) {
            updateNode(it->first, it->second * prob_miss_log, false);
        }
        for (BatchList::iterator it = occupied_cells.begin(); it != occupied_cells.end(); ++it) {
            updateNode(it->first, it->second * prob_hit_log, false);
        }
    }

    void ParallelBatchOcTree::insertPointCloudParallelBatch(const Pointcloud& pc, const point3d& origin, const int NUM_OF_THREADS)
    {
        if (pc.size() < 1 || NUM_OF_THREADS < 1)
            return;

        // Voxelize a given point clouds for fast distribution
        VoxPointclouds voxPC;
        for (unsigned int i = 0; i < pc.size(); ++i){
            voxPC[coordToKey(pc[i])].push_back(pc[i]);
        }

        // Sensor coordinate to Spherical coordinate
        SphPointcloud sphPC;
        int MAX_WORKLOADS = Sensor2Spherical(origin, voxPC, sphPC);

        // Workload balancing
        SphRefVector sphPCRef;
        for (unsigned int i = 0; i < sphPC.size(); i++)
            sphPCRef.push_back(&(sphPC[i]));
        std::queue<Task*> taskqueue;
        taskqueue.push(new Task(sphPCRef.begin(), sphPCRef.end(), MAX_WORKLOADS));

        while ((int)taskqueue.size() < NUM_OF_THREADS){
            // 0. Pop Queue
            Task* curTask = taskqueue.front();
            taskqueue.pop();

            // 1. Find split axis
            int axis = findSplitAxis(*curTask);

            // 2. Sort the points in spherical coordinate along with the split axis
            if (axis == 0)	std::sort(curTask->startIt, curTask->endIt, ComparePI());
            else 			std::sort(curTask->startIt, curTask->endIt, CompareTHETA());

            // 3. Find split point, which distribute rays to threads in balance
            SphRefVector::iterator splitIt = findSplitPoint(*curTask);

            // 4. Distribute Task
            taskqueue.push(new Task(curTask->startIt, splitIt, curTask->totalWorkload >> 1));
            taskqueue.push(new Task(splitIt, curTask->endIt, curTask->totalWorkload - (curTask->totalWorkload >> 1)));

            delete curTask;
        }

        // Initialize for parallal batching
        TaskSet set_of_tasks;
        while (!taskqueue.empty()){
            set_of_tasks.push_back(taskqueue.front());
            taskqueue.pop();
        }
        BatchList* set_of_free_cells = new BatchList[NUM_OF_THREADS];
        BatchList* set_of_occupied_cells = new BatchList[NUM_OF_THREADS];

        omp_set_num_threads(NUM_OF_THREADS);
#pragma omp parallel for
        for (int i = 0; i < (int)NUM_OF_THREADS; i++){
            unsigned int threadIdx = omp_get_thread_num();

            // Assign tasks to threads
            const Task* curTask = set_of_tasks[threadIdx];
            BatchList* free_cells = &set_of_free_cells[threadIdx];
            BatchList* occupied_cells = &set_of_occupied_cells[threadIdx];
            octomap::KeyRay* keyray = &(keyrays.at(threadIdx));

            // Process the ray tracing and batching
            for (SphRefVector::iterator it = curTask->startIt; it != curTask->endIt; it++){
                const std::vector<octomap::point3d>& pointlist = *((*it)->point);
                for (int j = 0; j < (int)pointlist.size(); j++){
                    const octomap::point3d& p = pointlist[j];

                    if (this->computeRayKeys(origin, p, *keyray)){
                        // Batch free cells
                        for (octomap::KeyRay::iterator it = keyray->begin(); it != keyray->end(); it++){
                            const BatchList::iterator& cell = free_cells->find(*it);
                            if (cell == free_cells->end())
                                free_cells->insert(std::pair<octomap::OcTreeKey, int>(*it, 1));
                            else
                                cell->second = cell->second + 1;
                        }
                        // Batch occupied cells
                        {
                            octomap::OcTreeKey key = coordToKey(p);
                            const BatchList::iterator& cell = occupied_cells->find(key);
                            if (cell == occupied_cells->end())
                                occupied_cells->insert(std::pair<octomap::OcTreeKey, int>(key, 1));
                            else
                                cell->second = cell->second + 1;
                        }
                    }
                }
            }

            delete curTask;
        }

        // insert data into tree  -----------------------
        for (int i = 0; i < NUM_OF_THREADS; i++){
            BatchList& free_cells = set_of_free_cells[i];
            for (BatchList::iterator it = free_cells.begin(); it != free_cells.end(); ++it) {
                updateNode(it->first, it->second * prob_miss_log, false);
            }
        }
        for (int i = 0; i < NUM_OF_THREADS; i++){
            BatchList& occupied_cells = set_of_occupied_cells[i];
            for (BatchList::iterator it = occupied_cells.begin(); it != occupied_cells.end(); ++it) {
                updateNode(it->first, it->second * prob_hit_log, false);
            }
        }

        delete[] set_of_free_cells;
        delete[] set_of_occupied_cells;
    }

    int ParallelBatchOcTree::Sensor2Spherical(const point3d& _origin, const VoxPointclouds& _voxPC, SphPointcloud& _sphPC)
    {
        int max_workloads = 0;
        octomap::OcTreeKey originKey = coordToKey(_origin);
        for (VoxPointclouds::const_iterator it = _voxPC.begin(); it != _voxPC.end(); ++it){
            octomap::OcTreeKey pointKey = coordToKey((it->second)[0]);
            octomap::point3d centerPoint = keyToCoord(pointKey);

            // "v" is the end point of a ray in sensor coordinate
            octomap::point3d v = centerPoint - _origin;
            double pi = atan2(v.y(), v.x());
            double theta = acos(v.z() / v.norm());

            int workloads = (int)it->second.size() * (abs(pointKey[0] - originKey[0]) + abs(pointKey[1] - originKey[1]) + abs(pointKey[2] - originKey[2]) + 2);
            max_workloads += workloads;

            _sphPC.push_back(SphPoint((point3d_collection*)(&(it->second)), pi, theta, workloads));
        }

        return max_workloads;
    }

    int ParallelBatchOcTree::findSplitAxis(const Task& _task)
    {
        double min[2] = { DBL_MAX, DBL_MAX };
        double max[2] = { -DBL_MAX, -DBL_MAX };
        for (SphRefVector::iterator it = _task.startIt; it != _task.endIt; ++it){
            const SphPoint& sp = *(*it);
            if (min[0] > sp.pi)		min[0] = sp.pi;
            if (min[1] > sp.theta)	min[1] = sp.theta;
            if (max[0] < sp.pi)		max[0] = sp.pi;
            if (max[1] < sp.theta)	max[1] = sp.theta;
        }

        if (max[0] - min[0] > max[1] - min[1])	return 0;	// Split Axis: PI
        else									return 1;	// Split Axis: Theta
    }

    ParallelBatchOcTree::SphRefVector::iterator ParallelBatchOcTree::findSplitPoint(const Task& _task)
    {
        const int TARGET_WORKLOADS = _task.totalWorkload >> 1;

        int workload = 0;
        for (SphRefVector::iterator it = _task.startIt; it != _task.endIt; ++it){
            workload += (*it)->workload;

            if (workload >= TARGET_WORKLOADS){
                if (abs(workload - (*it)->workload - _task.totalWorkload) < abs(workload - _task.totalWorkload))
                    return it;
                else
                    return (++it);
            }
        }

        return _task.startIt;
    }
}