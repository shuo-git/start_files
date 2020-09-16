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
ALPHA=1.0
BETA=0.25
SR=0.0
MP=0.7
GRAM=1t4
EXP=wmt14-en-de/cali-train/mse-a$ALPHA-b$BETA-sr$SR-mp$MP-gram$GRAM
#EXP=wmt14-en-de/cali-train/baseline
CHECKPOINT_DIR=$DISK2/exp/$EXP
mkdir -p $CHECKPOINT_DIR

OUTPUT_PATH=$DISK2/results/$EXP/evaluation
LOG_PATH=$DISK2/results/$EXP/logs

mkdir -p $OUTPUT_PATH
mkdir -p $LOG_PATH

echo 'Prepare valid data'
cp -r $DISK_DATA/$DATA/valid.de $DISK_DATA/$DATA/test.de $OUTPUT_PATH

RESTORE=$DISK2/exp/wmt14-en-de/wmt14_en_de_stanford_base/checkpoint_last.pt
cp ${RESTORE} $CHECKPOINT_DIR/checkpoint_last.pt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3.6 $DISK_CODE/train.py $DISK_DATA/$DATA/data-bin \
  --fp16 \
  -s en -t de \
  --lr 0.0007 --min-lr 1e-09 \
  --weight-decay 0.0 --clip-norm 0.0 --dropout 0.1 \
  --max-tokens 1024 \
  --update-freq 4 \
  --arch transformer \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-07 \
  --warmup-updates 4000 \
  --save-dir $CHECKPOINT_DIR \
  --tensorboard-logdir $LOG_PATH \
  --criterion label_smoothed_cross_entropy \
  --eps-pos 0.1 --eps-neg 0.7 --calibration-loss --mse-loss --shift-reg-weight $SR --mean-prob $MP \
  --alpha $ALPHA --beta $BETA --gram1 --gram2 --gram3 --gram4 \
  --no-progress-bar --log-format simple --log-interval 1 --quiet \
  --save-interval-updates 500 --keep-interval-updates 1000 \
  --max-update 200000 \
  --nbest 1 --beam 1 --lenpen 1.0  \
  --all-gather-list-size 522240 \
  --bleu-eval --valid-decoding-path $OUTPUT_PATH --multi-bleu-path $DISK_CODE/scripts \
  |& tee $LOG_PATH/train.log
