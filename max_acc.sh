#!/usr/bin/env bash

# place somewhere in system path to execute from cmd-line (e.g. ~/.local/bin)
# Synopsis: $ max_acc.sh <file>

FILE=${1}

MAX_DEV=$(jq '.' "$FILE" | grep 'Dev Acc' | sort | tail -n 1 | grep -Po '0\.\d+')

if [ -z "$MAX_DEV" ]
then
	echo "Old json file, grep manually..."
else
	jq '.' "$FILE" | grep -P -C 5 "Dev Accur\..: ${MAX_DEV},$"
fi

