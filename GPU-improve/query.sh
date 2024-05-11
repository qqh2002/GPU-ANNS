#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
	echo "Usage: $0 <cos/l2/ip> <data_name>" >&2
	echo "For example: $0 l2 sift" >&2
	exit 1
fi

dis=$1
name=$2

cd template

if [ "${dis}" = "l2" ]; then
	make query
elif [ "${dis}" = "cos" ]; then
	make query DISTTYPE=USE_COS_DIST_
elif [ "${dis}" = "ip" ]; then
	make query DISTTYPE=USE_IP_DIST_
else 
	echo "Usage: $0 <cos/l2/ip>" >&2
	echo "For example: $0 l2" >&2
	exit 1
fi

cp query ../${name}

cd ../${name}

total=0
for i in {1..3}
do
   return=$(./query)
   output=$(echo "$return" | awk '{print $3}')
   echo "$output"
   total=$((total+output))
done
averge=$((total / 3))
echo "avg times: $averge"