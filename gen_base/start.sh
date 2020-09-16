#!/usr/bin/env bash
export http_proxy="10.222.13.250:32810"
export https_proxy="10.222.13.250:32810"
export PYTHONIOENCODING=UTF-8

DISK1=/apdcephfs/private_vinceswang
DISK_DATA=$DISK1/DATASET
DISK_CODE=$DISK1/code/fairseq-T
DATA=wmt14_en_de_stanford
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
EXP=wmt14-en-de/base-cali-train-1

da=1.0
t=1.0
DECODE_PATH=$DISK_RESULTS/$EXP/inference
mkdir -p $DECODE_PATH

# for N in 10;do
# python3.6 $DISK_CODE/scripts/average_checkpoints.py --inputs $DISK_CKP/$EXP \
#   --output $DISK_CKP/$EXP/avg_last_${N}.pt \
#   --num-update-checkpoints $N
# done

for beam in 1 4;do
if [[ $beam = 1 ]]; then
  bsz=512
elif [[ $beam = 4 ]]; then
  bsz=128
elif [[ $beam = 100 ]]; then
  bsz=20
else
  bsz=2
fi

echo ${bsz}

# if [[ "${exp_i}" = "20-7" ]]; then
#   step=checkpoint1
# elif [[ "${exp_i}" = "20-2" ]]; then
#   step=checkpoint8
# elif [[ "${exp_i}" = "20-3" ]]; then
#   step=checkpoint7
# fi
#for step in {1..10};do
step=checkpoint_last
echo ${step}
CP=${step}.pt
CHECKPOINT=$DISK_CKP/$EXP/$CP
for SUBSET in valid test;do
#if [[ $beam = 4 ]] && [[ "$SUBSET" = "test" ]]; then
#    continue
#fi
GEN=${SUBSET}_${step}.${beam}.gen
echo "Evaluate on $DATA/$SUBSET with $CHECKPOINT"
CUDA_VISIBLE_DEVICES=0 python3.6 $DISK_CODE/generate.py \
  $DISK_DATA/$DATA/data-bin \
  --fp16 \
  -s en \
  -t de \
  --path $CHECKPOINT \
  --gen-subset $SUBSET \
  --lenpen ${da} \
  --beam ${beam} \
  --nbest 1 \
  --max-sentences ${bsz} \
  --temperature ${t} \
  > $DECODE_PATH/${GEN}

sh $DISK_CODE/scripts/compound_split_bleu.sh $DECODE_PATH/${GEN} | tee $DECODE_PATH/${GEN}.bleu

#grep ^H $DECODE_PATH/${GEN} | python3 $CALI/sorted_cut_fairseq_gen.py 2 > $DECODE_PATH/${GEN}.${beam}-best
done
done
