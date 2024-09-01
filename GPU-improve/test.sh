#!/usr/bin/env bash

cd sift

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