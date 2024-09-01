#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <cos/l2/ip>" >&2
	echo "For example: $0 cos" >&2
	exit 1
fi

dis=$1

mkdir build_template || true

cp *.h build_template
cp *.cc build_template
cp Makefile build_template

cd build_template

if [ "${dis}" = "cos" ]; then
	make anns_cpu
elif [ "${dis}" = "l2" ]; then
	make anns_cpu DISTTYPE=__USE_L2_DIST
elif [ "${dis}" = "ip" ]; then
	make anns_cpu DISTTYPE=__USE_IP_DIST
else
	echo "Usage: $0 <cos/l2/ip>" >&2
	echo "For example: $0 cos" >&2
	exit 1
fi

cp anns_cpu ..
