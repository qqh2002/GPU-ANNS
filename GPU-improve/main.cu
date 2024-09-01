#include<stdio.h>
#include<stdlib.h>
#include<string>
#include<set>
#include<cuda_runtime.h>
#include<fstream>
#include<sstream>
#include<iostream>
#include"parser.h"
#include"kernel_search.h"

float* data=NULL;
int* graph=NULL;
int* graph_=NULL;
int* result=NULL;
int*  cycle_times=NULL;
std::vector<float> query_vector;
float* query=NULL;
std::vector<std::vector<int>> cmp;

int num_vertices=PLACE_HOLDER_VERTICES;
int vertex_offset_shift=5;
int dim=PLACE_HOLDER_DIM;
int len_q=0;
int num_of_candidates=PLACE_HOLDER_CANDIDATES;
int num_of_topk=PLACE_HOLDER_DISPLAY;
int num_of_explored_points=0;
int num_of_topk_=0;
int width = PLACE_HOLDER_WIDTH;

void load_graph(std::string file = "bfsg.graph")
{
    FILE* fp = fopen(file.c_str(),"rb");
    auto cnt = fread(graph,sizeof(int) * (num_vertices << vertex_offset_shift),1,fp);
    fclose(fp);
}

void load_data(std::string file = "bfsg.data")
{
    FILE* fp = fopen(file.c_str(),"rb");
    auto cnt = fread(data,sizeof(float) * (num_vertices*dim),1,fp);
    fclose(fp);
}

void query_callback(int idx,std::vector<std::pair<int,float>> point){
    for(int i=0;i<point.size();i++)
    {
        query_vector.push_back(point[i].second);
    }
    len_q++;
}

void compute_recall(std::vector<std::vector<int>> compare)
{
    double recall;
    int ans=0,sum=0;
    std::ifstream file("ans.txt");
    std::vector<std::vector<int>> data;
    std::string line;

    if (!file.is_open()) {
        std::cout << "Unable to open file" << std::endl;
        return ;
    }

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<int> row;
        int value;

        while (iss >> value) {
            row.push_back(value);
        }

        data.push_back(row);
    }

    file.close();
    for(int i=0;i<compare.size()&&i<data.size();i++)
    {
        std::vector<int> r;
        std::vector<int> a;
        for(int j=0;j< num_of_topk;j++)
        {
            sum++;
            r.push_back(compare[i][j]);
            a.push_back(data[i][j]);
        }
        for(int j=0;j<r.size();j++)
        {
            for(int t=0;t<a.size();t++)
            {
                if(r[j]==a[t])
                {
                    ans++;
                }
            }
        }
    }
    recall=ans/double(sum);
    fprintf(stderr,"recall:%lf ans:%d sum:%d\n",recall,ans,sum);
}

int main()
{
    cudaSetDevice(0);
    graph=(int*)malloc(sizeof(int)*(num_vertices << vertex_offset_shift));
    data=(float*)malloc(sizeof(float)*(num_vertices * dim));
    load_graph();
    load_data();
    graph_=(int*)malloc(sizeof(int)*(num_vertices << vertex_offset_shift));
    for(int i = 0;i < num_vertices ;i++)
    {
        int len = graph[i<<vertex_offset_shift];
        for(int j = 0;j < (1 << vertex_offset_shift);j++)
        {
            if(j == 0)
            {
                graph_[i*(1<<vertex_offset_shift)+j]=num_vertices;
                continue;
            }
            if(j<=len)
            {
                graph_[i*(1<<vertex_offset_shift)+j] = graph[i*(1<<vertex_offset_shift)+j];
                
            }else{
                graph_[i*(1<<vertex_offset_shift)+j] = num_vertices;
            }
        }
    }
    free(graph);

    std::unique_ptr<Parser> query_parser(new Parser("test.txt",query_callback));
    query=(float*)malloc(sizeof(float)*(len_q*dim));
    for(int i=0;i<(len_q*dim);i++)
    {
        query[i]=query_vector[i];
    }

    num_of_topk_ = pow(2.0, ceil(log(num_of_topk) / log(2)));
    num_of_explored_points = num_of_candidates;
    num_of_candidates = pow(2.0, ceil(log(num_of_candidates) / log(2)));

    float* d_data;
    cudaMalloc(&d_data, sizeof(float) * num_vertices * dim);
    cudaMemcpy(d_data, data, sizeof(float) * num_vertices * dim, cudaMemcpyHostToDevice);

    float* d_query;
    cudaMalloc(&d_query, sizeof(float) * len_q * dim);
    cudaMemcpy(d_query, query, sizeof(float) * len_q * dim, cudaMemcpyHostToDevice);

    int* d_result;
    cudaMalloc(&d_result, sizeof(int) * len_q * num_of_topk_);
    result=(int*)malloc(sizeof(int) * len_q * num_of_topk_);

    int* d_graph;
    cudaMalloc(&d_graph, sizeof(int) * (num_vertices << vertex_offset_shift));
    cudaMemcpy(d_graph, graph_, sizeof(int) * (num_vertices << vertex_offset_shift), cudaMemcpyHostToDevice);

    int* d_cycle;
    cudaMalloc(&d_cycle,sizeof(int) * len_q);
    cycle_times = (int*)malloc(sizeof(int) * len_q);

    unsigned long long* h_time_breakdown;
    unsigned long long* d_time_breakdown;
    int num_of_phases = 6;
    cudaMallocHost(&h_time_breakdown, len_q * num_of_phases * sizeof(unsigned long long));
    cudaMalloc(&d_time_breakdown, len_q * num_of_phases * sizeof(unsigned long long));
    cudaMemset(d_time_breakdown, 0, len_q * num_of_phases * sizeof(unsigned long long));

    std::chrono::steady_clock::time_point kernel_begin = std::chrono::steady_clock::now();

    SearchDevice<<<len_q, 32, ((1 << vertex_offset_shift) * width + num_of_candidates) * (sizeof(KernelPair<float, int>) + sizeof(int))>>>(d_data, d_query, d_result, d_graph, num_vertices, 
                                                                                                                        len_q, vertex_offset_shift, num_of_candidates, num_of_topk_, 
                                                                                                                        num_of_explored_points, width, d_time_breakdown, d_cycle);
    
    cudaDeviceSynchronize();

	std::chrono::steady_clock::time_point kernel_end = std::chrono::steady_clock::now();
    printf("kernel takes %ld microseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_begin).count());                                                                                                             
    fprintf(stderr,"kernel takes %ld microseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_begin).count());                                                                                                             

    cudaMemcpy(result, d_result, sizeof(int) * len_q * num_of_topk_, cudaMemcpyDeviceToHost);
    cudaMemcpy(cycle_times, d_cycle, sizeof(int) * len_q, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_time_breakdown, d_time_breakdown, len_q * num_of_phases * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    unsigned long long stage_1 = 0;
	unsigned long long stage_2 = 0;
	unsigned long long stage_3 = 0;
	unsigned long long stage_4 = 0;
	unsigned long long stage_5 = 0;
	unsigned long long stage_6 = 0;

    for (int i = 0; i < len_q; i++) {
        stage_1	+= h_time_breakdown[i * num_of_phases];
        stage_2	+= h_time_breakdown[i * num_of_phases + 1];
        stage_3	+= h_time_breakdown[i * num_of_phases + 2];
        stage_4	+= h_time_breakdown[i * num_of_phases + 3];
        stage_5	+= h_time_breakdown[i * num_of_phases + 4];
        stage_6	+= h_time_breakdown[i * num_of_phases + 5];
    }

    unsigned long long sum_of_all_stages = stage_1 + stage_2 + stage_3 + stage_4 + stage_5 + stage_6;
    fprintf(stderr,"stages percentage: %lf %lf %lf\n",
                                       (double)(stage_1+stage_2) / sum_of_all_stages,
                                       (double)(stage_3) / sum_of_all_stages,
                                       (double)(stage_4+stage_5+stage_6) / sum_of_all_stages);
    // fprintf(stderr,"stages percentage: %lf %lf %lf %lf %lf %lf\n",
    //                                    (double)(stage_1) / sum_of_all_stages,
    //                                    (double)(stage_2) / sum_of_all_stages,
    //                                    (double)(stage_3) / sum_of_all_stages,
    //                                    (double)(stage_4) / sum_of_all_stages,
    //                                    (double)(stage_5) / sum_of_all_stages,
    //                                    (double)(stage_6) / sum_of_all_stages);
    // std::cout << "stages percentage: " << (double)(stage_1) / sum_of_all_stages << " "
    //                                 << (double)(stage_2) / sum_of_all_stages << " "
    //                                 << (double)(stage_3) / sum_of_all_stages << " "
    //                                 << (double)(stage_4) / sum_of_all_stages << " "
    //                                 << (double)(stage_5) / sum_of_all_stages << " "
    //                                 << (double)(stage_6) / sum_of_all_stages << std::endl;

    int sum_cycle = 0;
    for(int i=0;i<len_q;i++)
    {
        std::vector<int> tmp;
        sum_cycle += cycle_times[i];
        for(int j=0;j<num_of_topk;j++)
        {
            tmp.push_back(result[i*num_of_topk_+j]);
        }
        cmp.push_back(tmp);
    }
    fprintf(stderr,"avg_cycle_times:%f\n",sum_cycle/(len_q*1.0));
    compute_recall(cmp);
    free(graph_);
    free(data);
    free(query);
    free(result);
    return 0;
}