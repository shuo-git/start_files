#!/usr/bin/env bash
export http_proxy="10.222.13.250:32810"
export https_proxy="10.222.13.250:32810"
export PYTHONIOENCODING=UTF-8

DISK1=/apdcephfs/private_vinceswang
DISK_DATA=$DISK1/DATASET
DISK_CODE=$DISK1/code/fairseq-T
DATA=wmt14_en_de_stanford

pip list | grep fairseq
if [ $? != 0 ]; then
  echo 'Install Fairseq First'
  cd $DISK_CODE
  pip install --editable .
fi

DISK2=/apdcephfs/share_916081/vinceswang
DISK_CKP=$DISK2/exp
DISK_RESULTS=$DISK2/results
EXP=${DATA}_big
# CP=checkpoint_best.pt
# CHECKPOINT=$DISK_CKP/$EXP/$CP

DECODE_PATH=$DISK_RESULTS/$EXP/inference
mkdir -p $DECODE_PATH

for N in 10 20 30;do
python3.6 $DISK_CODE/scripts/average_checkpoints.py --inputs $DISK_CKP/$EXP \
  --output $DISK_CKP/$EXP/avg_last_${N}.pt \
  --num-update-checkpoints $N
done

for CP in avg_last_10.pt avg_last_20.pt avg_last_30.pt;do
CHECKPOINT=$DISK_CKP/$EXP/$CP

for SUBSET in valid test;do

echo "Evaluate on $DATA/$SUBSET with $CHECKPOINT"
CUDA_VISIBLE_DEVICES=0 python3.6 $DISK_CODE/generate.py \
  $DISK_DATA/$DATA/data-bin \
  --fp16 \
  -s en \
  -t de \
  --path $CHECKPOINT \
  --gen-subset $SUBSET \
  --lenpen 0.6 \
  --beam 4 \
  --remove-bpe \
  --decoding-path $DECODE_PATH \
  --num-ref $DATASET=1 \
  --multi-bleu-path $DISK_CODE/scripts/ \
  --valid-decoding-path $DECODE_PATH \
  > $DECODE_PATH/${SUBSET}_$CP.gen

sh $DISK_CODE/scripts/compound_split_bleu.sh $DECODE_PATH/${SUBSET}_$CP.gen

done
done
