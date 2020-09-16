#!/usr/bin/env bash
export http_proxy="10.222.13.250:32810"
export https_proxy="10.222.13.250:32810"
export PYTHONIOENCODING=UTF-8

DISK1=/apdcephfs/private_vinceswang
DISK2=/apdcephfs/share_916081/vinceswang
CALI=$DISK1/code/Cali-Ana
DATA=$DISK2/results/wmt14-en-de/wmt14_en_de_stanford_base/inference-bak
#echo "hello"
python3 $CALI/beam_oracle_analysis.py --hyp $DATA/train.search.100.gen.top-100 --out $DATA/train.search.100.gen.top-100.top-1 --beam 100
