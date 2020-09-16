#!/usr/bin/env bash
export http_proxy="10.222.13.250:32810"
export https_proxy="10.222.13.250:32810"
export PYTHONIOENCODING=UTF-8

DISK1=/apdcephfs/private_vinceswang
DISK_DATA=$DISK1/DATASET
DISK_CODE=$DISK1/code/fairseq-T
DATA=wmt14_en_de_stanford_sampled/100w
CALI=$DISK1/code/Cali-Ana

pip list | grep fairseq
if [ $? != 0 ]; then
  echo 'Install Fairseq First'
  cd $DISK_CODE
  pip install --editable .
fi

DISK2=/apdcephfs/share_916081/vinceswang
DISK_CKP=$DISK2/exp
DISK_RESULTS=$DISK2/results
EXP=wmt14-en-de/wmt14_en_de_stanford_base
DECODE_PATH=$DISK_RESULTS/$EXP/inference
mkdir -p $DECODE_PATH

beam=-1

if [[ $beam = 1 ]]; then
  bsz=512
elif [[ $beam = 4 ]]; then
  bsz=128
elif [[ $beam = 100 ]]; then
  bsz=20
else
  bsz=128
fi

bsz=128

echo ${bsz}

CP=avg_last_10.pt
CHECKPOINT=$DISK_CKP/$EXP/$CP
SUBSET=train

GEN=${SUBSET}.sampling.gen
echo "Evaluate on $DATA/$SUBSET with $EXP/$CP"
CUDA_VISIBLE_DEVICES=0 python3.6 $DISK_CODE/generate.py \
  $DISK_DATA/$DATA/data-bin \
  --fp16 \
  -s en \
  -t de \
  --path $CHECKPOINT \
  --gen-subset $SUBSET \
  --lenpen 1.0 \
  --max-sentences ${bsz} \
  --temperature 1.0 \
  --sampling \
  --sampling-topk ${beam} \
  --nbest 1 \
  --beam 1 \
  > $DECODE_PATH/$GEN

# --beam ${beam} \
# --nbest ${beam} \
# --sampling \
# --sampling-topk ${beam} \

grep ^H $DECODE_PATH/${GEN} | python3 $CALI/sorted_cut_fairseq_gen.py 2 > $DECODE_PATH/${GEN}.sys