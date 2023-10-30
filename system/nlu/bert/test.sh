#!/bin/bash

train_script_path="$HOME/ConvLab-2/convlab2/nlu/jointBERT"
# train_script_path="/data/group1/z44383r/dev/ppn-nlg/ConvLab-2/convlab2/nlu/jointBERT"

current_dpath=$(cd $(dirname $0);pwd)
config_path="${current_dpath}/$1"

# Run test script
python ${train_script_path}/test.py --config_path ${config_path}
