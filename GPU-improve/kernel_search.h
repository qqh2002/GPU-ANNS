#pragma once

#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<cuda_runtime.h>
#include<chrono>
#include<iostream>

#define FULL_MASK 0xffffffff
#define Max 0x1fffffff
#define DIM PLACE_HOLDER_DIM
#define WIDTH PLACE_HOLDER_WIDTH

template<class A,class B>
struct KernelPair{
    A first;
    B second;
	
	__device__
	KernelPair(){}


	__device__
    bool operator <(KernelPair& kp) const{
        return first < kp.first;
    }


	__device__
    bool operator >(KernelPair& kp) const{
        return first > kp.first;
    }
};


__global__
void SearchDevice(float* d_data, float* d_query, int* d_result, int* d_graph, int total_num_of_points, int num_of_query_points, int offset_shift, 
                    int num_of_candidates, int num_of_results, int num_of_explored_points, int width, unsigned long long* time_breakdown, int* d_cycle) {
	int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int size_of_warp = 32;

    extern __shared__ KernelPair<float, int> shared_memory_space_s[];
    KernelPair<float, int>* neighbors_array = shared_memory_space_s;
    int* flags = (int*)(shared_memory_space_s + num_of_candidates + (1 << offset_shift) * width);
    
    int crt_point_id = b_id;
    int crt_cycle = 0;
    int* crt_result = d_result + crt_point_id * num_of_results;
    unsigned long long* crt_time_breakdown = time_breakdown + crt_point_id * 6;

DECLARE_QUERY_POINT_

    int step_id;
    int substep_id;

    int num_of_visited_points_one_batch = 1 << offset_shift;
    int length_of_compared_list = num_of_candidates;
    if(num_of_visited_points_one_batch * width < num_of_candidates){
        length_of_compared_list = num_of_visited_points_one_batch * width;
    }
    
    int flag_all_blocks = 1;

    int temporary_flag;
    int first_position_of_flag[WIDTH];
    memset(first_position_of_flag, 0 ,sizeof(first_position_of_flag));
    KernelPair<float, int> temporary_neighbor;

    for (int i = 0; i < (num_of_candidates + num_of_visited_points_one_batch * width + size_of_warp - 1) / size_of_warp; i++) {
        int unrollt_id = t_id + size_of_warp * i;

        if (unrollt_id < num_of_candidates + num_of_visited_points_one_batch * width) {
            flags[unrollt_id] = 0;

            neighbors_array[unrollt_id].first = Max;
            neighbors_array[unrollt_id].second = total_num_of_points;
        }
    }

    if (t_id == 0) {
        neighbors_array[0].second = 0;
        flags[0] = 1;
    }

    __syncthreads();

    int target_point_id = 0;
    
DECLARE_SECOND_FEATURE_

COMPUTATION_

SUM_UP_ 

    __syncthreads();

WITHIN_WARP_
            
    if (t_id == 0) {
        neighbors_array[0].first = dist;
    }

    __syncthreads();
   	
    int w = 1;

    while (flag_all_blocks) {
        
        crt_cycle++;

        if (t_id == 0) {
            flags[first_position_of_flag[0]] = 0;
        }

        __syncthreads();

        auto stage1_start = clock64();

        while(w < width && w < crt_cycle)
        {
            int  f = 1;
            for (int i = 0; i < (num_of_explored_points + size_of_warp - 1) / size_of_warp; i++) {
                int unrollt_id = t_id + size_of_warp * i;
                int crt_flag = 0;

                if(unrollt_id < num_of_explored_points){
                    crt_flag = flags[unrollt_id];
                }

                first_position_of_flag[w] = __ballot_sync(FULL_MASK, crt_flag);

                if(first_position_of_flag[w] != 0){
                    first_position_of_flag[w] = size_of_warp * i + __ffs(first_position_of_flag[w]) - 1;
                    if(t_id == 0)
                    {
                        flags[first_position_of_flag[w]] = 0;
                    }
                    __syncthreads();
                    w++;
                    break;
                }else if(i == (num_of_explored_points + size_of_warp - 1) / size_of_warp - 1){
                    f = 0;
                }
            }
            
            if(f == 0)
            {
                break;
            }
        }

        auto stage1_end = clock64();
        if(t_id == 0){
            crt_time_breakdown[0] += stage1_end - stage1_start;
        }

        __syncthreads();

        auto stage2_start = clock64();

        for(int k = 0; k < width; k++)
        {
            auto offset = neighbors_array[first_position_of_flag[k]].second << offset_shift;
        
            if(k < w){
                for (int i = 0; i < (num_of_visited_points_one_batch + size_of_warp - 1) / size_of_warp; i++) {
                    int unrollt_id = t_id + size_of_warp * i;

                    if (unrollt_id < num_of_visited_points_one_batch) {
                        neighbors_array[num_of_candidates + num_of_visited_points_one_batch * k + unrollt_id].second = (d_graph + offset)[unrollt_id];
                        flags[num_of_candidates + num_of_visited_points_one_batch * k + unrollt_id] = 1;
                    }
                }
            }else{
                for (int i = 0; i < (num_of_visited_points_one_batch + size_of_warp - 1) / size_of_warp; i++) {
                    int unrollt_id = t_id + size_of_warp * i;

                    if (unrollt_id < num_of_visited_points_one_batch) {
                        neighbors_array[num_of_candidates + num_of_visited_points_one_batch * k + unrollt_id].second = total_num_of_points;
                        neighbors_array[num_of_candidates + num_of_visited_points_one_batch * k + unrollt_id].first = Max;
                        flags[num_of_candidates + num_of_visited_points_one_batch * k + unrollt_id] = 1;
                    }
                }
            }
        }

        auto stage2_end = clock64();
        if(t_id == 0){
            crt_time_breakdown[1] += stage2_end - stage2_start;
        }


        __syncthreads();

        auto stage3_start = clock64();

        for (int i = 0; i < num_of_visited_points_one_batch * w; i++) {
            int target_point_id = neighbors_array[num_of_candidates + i].second;
            
            if (target_point_id >= total_num_of_points) {
                if(t_id == 0)
                    neighbors_array[num_of_candidates + i].first = Max;
                __syncthreads();
                continue;
            }
            

DECLARE_SECOND_FEATURE_

COMPUTATION_

SUM_UP_
        
            __syncthreads();


WITHIN_WARP_

            
            if (t_id == 0) {
                neighbors_array[num_of_candidates+i].first = dist;
            }

            __syncthreads();

        }

        auto stage3_end = clock64();
        if(t_id == 0){
            crt_time_breakdown[2] += stage3_end - stage3_start;
        }

        __syncthreads();

        auto stage4_start = clock64();

        for (int temparory_id = 0; temparory_id < (num_of_visited_points_one_batch * w + size_of_warp - 1) / size_of_warp; temparory_id++) {
            int unrollt_id = t_id + size_of_warp * temparory_id;
            if (unrollt_id < num_of_visited_points_one_batch * w) {
                float target_distance = neighbors_array[num_of_candidates+unrollt_id].first;
                int flag_of_find = -1;
                int low_end = 0;
                int high_end = num_of_candidates - 1;
                int middle_end;
                while (low_end <= high_end) {
                    middle_end = (high_end + low_end) / 2;
                    if (target_distance == neighbors_array[middle_end].first) {
                        if (middle_end > 0 && neighbors_array[middle_end - 1].first == neighbors_array[middle_end].first) {
                            high_end = middle_end - 1;
                        } else {
                            flag_of_find = middle_end;
                            break;
                        }
                    } else if (target_distance < neighbors_array[middle_end].first) {
                        high_end = middle_end - 1;
                    } else {
                        low_end = middle_end + 1;
                    }
                }
                if (flag_of_find != -1) {
                    if (neighbors_array[num_of_candidates + unrollt_id].second == neighbors_array[flag_of_find].second) {
                        neighbors_array[num_of_candidates + unrollt_id].first = Max;
                    } else {
                        int position_of_find_element = flag_of_find + 1;

                        while (neighbors_array[position_of_find_element].first == neighbors_array[num_of_candidates + unrollt_id].first) {
                            if (neighbors_array[num_of_candidates + unrollt_id].second == neighbors_array[position_of_find_element].second) {
                                neighbors_array[num_of_candidates + unrollt_id].first = Max;
                                break;
                            }
                            position_of_find_element++;
                        }
                    }
                }
            }
        }

        auto stage4_end = clock64();
        if(t_id == 0){
            crt_time_breakdown[3] += stage4_end - stage4_start;
        }

        __syncthreads();

        auto stage5_start = clock64();

        step_id = 1;
        substep_id = 1;

        for (; step_id <= (num_of_visited_points_one_batch * width) / 2; step_id *= 2) {
            substep_id = step_id;

            for (; substep_id >= 1; substep_id /= 2) {
                for (int temparory_id = 0; temparory_id < ((num_of_visited_points_one_batch * width)/2+size_of_warp-1) / size_of_warp; temparory_id++) {
                    int unrollt_id = num_of_candidates + ((t_id + size_of_warp * temparory_id) / substep_id) * 2 * substep_id + ((t_id + size_of_warp * temparory_id) & (substep_id - 1));
                    
                    if (unrollt_id < num_of_candidates + (num_of_visited_points_one_batch * width)) {
                        if (((t_id + size_of_warp * temparory_id) / step_id) % 2 == 0) {
                            if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                                temporary_neighbor = neighbors_array[unrollt_id];
                                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                                neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            }
                        } else {
                            if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                                temporary_neighbor = neighbors_array[unrollt_id];
                                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                                neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            }
                        }
                    }
                }
            }
        }

        auto stage5_end = clock64();
        if(t_id == 0){
            crt_time_breakdown[4] += stage5_end - stage5_start;
        }

        __syncthreads();

        auto stage6_start = clock64();

        for (int temparory_id = 0; temparory_id < (length_of_compared_list + size_of_warp - 1) / size_of_warp; temparory_id++) {
            int unrollt_id = num_of_candidates - length_of_compared_list + t_id + size_of_warp * temparory_id;
            if (unrollt_id < num_of_candidates) {
                if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + num_of_visited_points_one_batch * width].first) {
                    temporary_neighbor = neighbors_array[unrollt_id];
                    neighbors_array[unrollt_id] = neighbors_array[unrollt_id + num_of_visited_points_one_batch * width];
                    neighbors_array[unrollt_id + num_of_visited_points_one_batch * width] = temporary_neighbor;
                    
                    temporary_flag = flags[unrollt_id];
                    flags[unrollt_id] = flags[unrollt_id + num_of_visited_points_one_batch * width];
                    flags[unrollt_id + num_of_visited_points_one_batch * width] = temporary_flag;
                }
            }
        }

        step_id = num_of_candidates / 2;
        substep_id = num_of_candidates / 2;
        for (; substep_id >= 1; substep_id /= 2) {
            for (int temparory_id = 0; temparory_id < (num_of_candidates / 2 + size_of_warp - 1) / size_of_warp; temparory_id++) {
                int unrollt_id = ((t_id + size_of_warp * temparory_id)/ substep_id) * 2 * substep_id + ((t_id + size_of_warp * temparory_id) & (substep_id - 1));
                if (unrollt_id < num_of_candidates) {
                    if (((t_id + size_of_warp * temparory_id) / step_id) % 2 == 0) {
                        if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                            temporary_neighbor = neighbors_array[unrollt_id];
                            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                            neighbors_array[unrollt_id + substep_id] = temporary_neighbor;

                            temporary_flag = flags[unrollt_id];
                            flags[unrollt_id] = flags[unrollt_id + substep_id];
                            flags[unrollt_id + substep_id] = temporary_flag;
                        }
                    } else {
                        if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                            temporary_neighbor = neighbors_array[unrollt_id];
                            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                            neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            
                            temporary_flag = flags[unrollt_id];
                            flags[unrollt_id] = flags[unrollt_id + substep_id];
                            flags[unrollt_id + substep_id] = temporary_flag;
                        }
                    }
                }
            }
        }

        auto stage6_end = clock64();
        if(t_id == 0){
            crt_time_breakdown[5] += stage6_end - stage6_start;
        }

        __syncthreads();

        stage1_start = clock64();

        w = 0;

        for (int i = 0; i < (num_of_explored_points + size_of_warp - 1) / size_of_warp; i++) {
            int unrollt_id = t_id + size_of_warp * i;
            int crt_flag = 0;

            if(unrollt_id < num_of_explored_points){
                crt_flag = flags[unrollt_id];
            }
            first_position_of_flag[0] = __ballot_sync(FULL_MASK, crt_flag);

            if(first_position_of_flag[0] != 0){
                first_position_of_flag[0] = size_of_warp * i + __ffs(first_position_of_flag[0]) - 1;
                w++;
                break;
            }else if(i == (num_of_explored_points + size_of_warp - 1) / size_of_warp - 1){
                flag_all_blocks = 0;
            }
        }

        stage1_end = clock64();
        if(t_id == 0){
            crt_time_breakdown[0] += stage1_end - stage1_start;
        }

        __syncthreads();
    }

    for (int i = 0; i < (num_of_results + size_of_warp - 1) / size_of_warp; i++) {
        int unrollt_id = t_id + size_of_warp * i;
    
        if (unrollt_id < num_of_results) {
            crt_result[unrollt_id] = neighbors_array[unrollt_id].second;
        }
    }

    if(t_id == 0)
    {
        d_cycle[b_id] = crt_cycle;
    }
    
    __syncthreads();

}