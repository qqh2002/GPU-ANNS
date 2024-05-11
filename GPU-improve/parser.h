#pragma once

#include<stdlib.h>
#include<memory>
#include<vector>
#include<functional>



class Parser{
private:
	const int ONE_BASED_LIBSVM = 1;
    const int MAX_LINE = 10000000;
    std::function<void(int,std::vector<std::pair<int,float>>)> consume;

    std::vector<int> tokenize(char* buff){
        std::vector<int> ret;
        int i = 0;
        while(*(buff + i) != '\0'){
            if(*(buff + i) == ':' || *(buff + i) == ' ')
                ret.push_back(i);
            ++i;
        }
        return ret;
    }

    std::vector<std::pair<int,float>> parse(std::vector<int> tokens,char* buff){
        std::vector<std::pair<int,float>> ret;
        ret.reserve(tokens.size() / 2);
        for(int i = 0;i + 1 < tokens.size();i += 2){
            int index;
            float val;
            sscanf(buff + tokens[i] + 1,"%d",&index);
			index -= ONE_BASED_LIBSVM;
			double tmp;
            sscanf(buff + tokens[i + 1] + 1,"%lf",&tmp);
			val = tmp;
            ret.push_back(std::make_pair(index,val));
        }
        return ret;
    }


public:

    Parser(const char* path,std::function<void(int,std::vector<std::pair<int,float>>)> consume) : consume(consume){
        auto fp = fopen(path,"r");
        if(fp == NULL){
            exit(1);
        }
        std::unique_ptr<char[]> buff(new char[MAX_LINE]);
        std::vector<std::string> buffers;
        int idx = 0;
        while(fgets(buff.get(),MAX_LINE,fp)){
            auto tokens = tokenize(buff.get());
            auto values = parse(tokens,buff.get());
            consume(idx,values);
            ++idx;
        }
        fclose(fp);
    }
    
};
