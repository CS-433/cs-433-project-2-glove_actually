#!/bin/bash

# Note that this script uses GNU-style sed as gsed. On Mac OS, you are required to first https://brew.sh/
#    brew install gnu-sed
# on linux, use sed instead of gsed in the command below:
cat $1 | gsed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > ./glove/tmp/vocab_full.txt
