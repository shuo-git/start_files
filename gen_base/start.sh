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
EXP=${DATA}_base_reduce_inference_ece-1
DECODE_PATH=$DISK_RESULTS/$EXP/inference
mkdir -p $DECODE_PATH

# for N in 5 10;do
# python3.6 $DISK_CODE/scripts/average_checkpoints.py --inputs $DISK_CKP/$EXP \
#   --output $DISK_CKP/$EXP/avg_last_${N}.pt \
#   --num-update-checkpoints $N
# done

for beam in 1 10 100;do
for step in checkpoint1 checkpoint2 checkpoint3 checkpoint_last;do
echo ${step}
CP=${step}.pt
CHECKPOINT=$DISK_CKP/$EXP/$CP
for SUBSET in test;do
GEN=${SUBSET}_${step}.${beam}.gen
echo "Evaluate on $DATA/$SUBSET with $CHECKPOINT"
CUDA_VISIBLE_DEVICES=0 python3.6 $DISK_CODE/generate.py \
  $DISK_DATA/$DATA/data-bin \
  --fp16 \
  -s en \
  -t de \
  --path $CHECKPOINT \
  --gen-subset $SUBSET \
  --lenpen 0.6 \
  --beam ${beam} \
  --max-sentences 20 \
  > $DECODE_PATH/${GEN}

sh $DISK_CODE/scripts/compound_split_bleu.sh $DECODE_PATH/${GEN}

done
done
done
