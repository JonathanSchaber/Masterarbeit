#!/usr/bin/env bash

FILE=${1}

MAX_DEV=$(jq '.' "$FILE" | grep 'Dev Acc' | sort | tail -n 1 | grep -Po '0\.\d+')

if [ -z "$MAX_DEV" ]
then
	echo "Old json file, grep manually..."
else
	jq '.' "$FILE" | grep -PA 1 ${MAX_DEV}
fi
	

