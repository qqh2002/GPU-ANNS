#!/usr/bin/env bash

if [ "$#" -ne 5 ]; then
	echo "Usage: $0 <pq_size> <dim> <result_num> <num_vertices> <num_width>" >&2
	echo "For example: $0 100 128 10 1000000 1" >&2
	exit 1
fi

topk=$1
dim=$2
display=$3
num_vertices=$4
num_width=$5

DIR="template"

if [ -d "$DIR" ]; then
    rm -rf ${DIR}
fi

mkdir template || true

cp *.h template
cp *.cu template
cp -R macro template
cp Makefile template

cd template

nl='\n'

while IFS= read -r line || [[ -n "$line" ]]; do
	line_with_tail_symbol="${line}${nl}DECLARE_QUERY_POINT_"
    sed -i "s@DECLARE_QUERY_POINT_@${line_with_tail_symbol}@" *.h
done < macro/declare_query_point.h
sed -i "s@DECLARE_QUERY_POINT_@@" *.h


while IFS= read -r line || [[ -n "$line" ]]; do
	line_with_tail_symbol="${line}${nl}DECLARE_SECOND_FEATURE_"
    sed -i "s@DECLARE_SECOND_FEATURE_@${line_with_tail_symbol}@" *.h
done < macro/declare_second_feature.h
sed -i "s@DECLARE_SECOND_FEATURE_@@" *.h


while IFS= read -r line || [[ -n "$line" ]]; do
	line_with_tail_symbol="${line}${nl}COMPUTATION_"
    sed -i "s@COMPUTATION_@${line_with_tail_symbol}@" *.h
done < macro/computation.h
sed -i "s@COMPUTATION_@@" *.h


while IFS= read -r line || [[ -n "$line" ]]; do
	line_with_tail_symbol="${line}${nl}SUM_UP_"
    sed -i "s@SUM_UP_@${line_with_tail_symbol}@" *.h
done < macro/sum_up.h
sed -i "s@SUM_UP_@@" *.h


while IFS= read -r line || [[ -n "$line" ]]; do
	line_with_tail_symbol="${line}${nl}WITHIN_WARP_"
    sed -i "s@WITHIN_WARP_@${line_with_tail_symbol}@" *.h
done < macro/within_warp.h
sed -i "s@WITHIN_WARP_@@" *.h


sed -i "s/PLACE_HOLDER_CANDIDATES/${topk}/g" main.cu
sed -i "s/PLACE_HOLDER_DIM/${dim}/g" kernel_search.h
sed -i "s/PLACE_HOLDER_DIM/${dim}/g" main.cu
sed -i "s/PLACE_HOLDER_DISPLAY/${display}/g" main.cu
sed -i "s/PLACE_HOLDER_VERTICES/${num_vertices}/g" main.cu
sed -i "s/PLACE_HOLDER_WIDTH/${num_width}/g" main.cu
sed -i "s/PLACE_HOLDER_WIDTH/${num_width}/g" kernel_search.h