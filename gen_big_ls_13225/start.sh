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
EXP=${DATA}_big_ls_13225
DECODE_PATH=$DISK_RESULTS/$EXP/inference
mkdir -p $DECODE_PATH

N=10
# python3.6 $DISK_CODE/scripts/average_checkpoints.py --inputs $DISK_CKP/$EXP \
#   --output $DISK_CKP/$EXP/avg_last_${N}.pt \
#   --num-update-checkpoints $N

# for step in {2000..300000..2000};do
# CP=checkpoint_*_${step}.pt
for beam in 200 400 800;do
for CP in avg_last_10.pt;do
CHECKPOINT=$DISK_CKP/$EXP/$CP
for SUBSET in valid test;do
GEN=$DECODE_PATH/${SUBSET}_${CP}.${beam}.gen
echo "Evaluate on $DATA/$SUBSET with $CHECKPOINT, beam size = ${beam}"
CUDA_VISIBLE_DEVICES=0 python3.6 $DISK_CODE/generate.py \
  $DISK_DATA/$DATA/data-bin \
  --fp16 \
  -s en \
  -t de \
  --path $CHECKPOINT \
  --gen-subset $SUBSET \
  --lenpen 0.6 \
  --beam ${beam} \
  --max-sentences 1 \
  --decoding-path $DECODE_PATH \
  --num-ref $DATASET=1 \
  --multi-bleu-path $DISK_CODE/scripts/ \
  --valid-decoding-path $DECODE_PATH \
  > $GEN

sh $DISK_CODE/scripts/compound_split_bleu.sh $GEN

done
done
done
