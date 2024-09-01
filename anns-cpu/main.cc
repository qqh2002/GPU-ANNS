#include<stdio.h>
#include<string.h>
#include <iostream>
#include <fstream>
#include<sstream>
#include <queue>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include"logger.h"
#include"parser_dense.h"
#include"parser.h"
#include"data.h"
#include"graph.h"
#include<stdlib.h>
#include<memory>
#include<vector>
#include<functional>
#include"bithash.h"

std::unique_ptr<Data> data;
std::unique_ptr<GraphWrapper> graph; 
int topk = 0;
int display_topk = 1;
BitHash bithash;
std::vector<std::vector<int>> cmp;

void build_callback(idx_t idx,std::vector<std::pair<int,value_t>> point){
	#ifdef __ENABLE_HASH
	point = bithash.hash2kv(point);
	#endif
    data->add(idx,point);
    graph->add_vertex(idx,point);
}

void query_callback(idx_t idx,std::vector<std::pair<int,value_t>> point){
	#ifdef __ENABLE_HASH
	point = bithash.hash2kv(point);
	#endif
    std::vector<idx_t> result;
    graph->search_top_k(point,topk,result);
    cmp.push_back(result);
    // for(int i = 0;i < result.size() && i < display_topk;++i)
    //     printf("%d ",result[i]);
    //printf("\n");
}

void compute_recall(std::vector<std::vector<int>> compare)
{
    double recall;
    int ans=0,sum=0;
    std::ifstream file("ans.txt");
    std::vector<std::vector<int>> data;
    std::string line;

    // 检查文件是否成功打开
    if (!file.is_open()) {
        std::cout << "Unable to open file" << std::endl;
        return ;
    }

    // 逐行读取文件
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<int> row;
        int value;

        // 分割每一行的内容
        while (iss >> value) {
            row.push_back(value);
        }

        // 将这一行的数据添加到二维数组中
        data.push_back(row);
    }
    // 关闭文件
    file.close();
    for(int i=0;i<compare.size()&&i<data.size();i++)
    {
        std::vector<int> r;
        std::vector<int> a;
        for(int j=0;j< 10;j++)
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
    printf("recall:%lf ans:%d sum:%d\n",recall,ans,sum);
}

void usage(char** argv){
    printf("Usage: %s <build/test> <build_data> <query_data> <search_top_k> <row> <dim> <return_top_k> <l2/ip/cos>\n",argv[0]);
}

int main(int argc,char** argv){
    if(argc != 9){
        usage(argv);
        return 1;
    }
	size_t row = atoll(argv[5]);
	int dim = atoi(argv[6]);
	display_topk = atoi(argv[7]);
	std::string dist_type = argv[8];
	
	#ifdef __ENABLE_HASH
	const int HASH_DIM = 256;
	bithash = BitHash(dim,HASH_DIM);
	int old_dim = dim;
	dim = HASH_DIM / sizeof(value_t) / 8;
	dist_type = "hash";
	#endif

	data = std::unique_ptr<Data>(new Data(row,dim));


	if(dist_type == "l2"){
		graph = std::unique_ptr<GraphWrapper>(new FixedDegreeGraph<0>(data.get())); 
	}else if(dist_type == "ip"){
		graph = std::unique_ptr<GraphWrapper>(new FixedDegreeGraph<1>(data.get())); 
	}else if(dist_type == "cos"){
		graph = std::unique_ptr<GraphWrapper>(new FixedDegreeGraph<2>(data.get())); 
	}else if(dist_type == "hash"){
		graph = std::unique_ptr<GraphWrapper>(new FixedDegreeGraph<3>(data.get())); 
	}else{
		usage(argv);
		return 1;
	}
    std::string mode = std::string(argv[1]);
    topk = atoi(argv[4]);
    if(mode == "build"){
        std::unique_ptr<Parser> build_parser(new Parser(argv[2],build_callback));
        fprintf(stderr,"Writing the graph and data...");    
        data->dump();
        fprintf(stderr,"...");    
        graph->dump();
        fprintf(stderr,"done\n");    
    }else if(mode == "test"){
        fprintf(stderr,"Loading the graph and data...");    
        data->load();
        fprintf(stderr,"...");    
        graph->load();
        fprintf(stderr,"done\n");
        std::chrono::steady_clock::time_point kernel_begin = std::chrono::steady_clock::now();    
        std::unique_ptr<Parser> query_parser(new Parser(argv[3],query_callback));
        std::chrono::steady_clock::time_point kernel_end = std::chrono::steady_clock::now();
        fprintf(stderr,"kernel takes %ld microseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_begin).count());
        compute_recall(cmp);
    }else{
        usage(argv);
        return 1;
    }


    return 0;
}
