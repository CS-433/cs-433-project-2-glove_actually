#!/bin/bash

# Note that this script uses GNU-style sed as gsed. On Mac OS, you are required to first https://brew.sh/
#    brew install gnu-sed
# on linux, use sed instead of gsed in the command below:
cat data/train_pos_full.txt data/train_neg_full.txt | gsed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab_full.txt
