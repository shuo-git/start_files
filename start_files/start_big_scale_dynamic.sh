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
EXP=${DATA}_big_scale_dynamic

CHECKPOINT_DIR=$DISK2/exp/$EXP
mkdir -p $CHECKPOINT_DIR

OUTPUT_PATH=$DISK2/results/$EXP/evaluation
LOG_PATH=$DISK2/results/$EXP/logs

if ! [ -d $DISK2/results/$EXP ]; then
  echo "results/$EXP not exist"
  mkdir -p $OUTPUT_PATH
  mkdir -p $LOG_PATH
else
  echo "results/$EXP exist, will be cleaned"
  rm -r $DISK2/results/$EXP
  mkdir -p $OUTPUT_PATH
  mkdir -p $LOG_PATH
fi

echo 'Prepare valid data'
cp -r $DISK_DATA/$DATA/valid.de $DISK_DATA/$DATA/test.de $OUTPUT_PATH
sed -i -e 's/@@ //g' $OUTPUT_PATH/valid.de
sed -i -e 's/@@ //g' $OUTPUT_PATH/test.de

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3.6 $DISK_CODE/train.py $DISK_DATA/$DATA/data-bin \
  --fp16 \
  -s en -t de \
  --lr 1e-07 --min-lr 1e-09 \
  --weight-decay 0.0 --clip-norm 0.0 \
  --max-tokens 3584 \
  --update-freq 16 \
  --arch transformer_vaswani_wmt_en_de_big \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --ddp-backend=no_c10d \
  --lr-scheduler cosine --warmup-init-lr 1e-07 --warmup-updates 10000 \
  --lr-shrink 1 --max-lr 0.001 \
  --t-mult 1 --lr-period-updates 20000 \
  --save-dir $CHECKPOINT_DIR \
  --tensorboard-logdir $LOG_PATH \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --no-progress-bar \
  --log-format simple \
  --log-interval 1 \
  --save-interval-updates 2000 \
  --max-update 30000 \
  --beam 1 \
  --remove-bpe \
  --quiet \
  --all-gather-list-size 522240 \
  --num-ref $DATA=1 \
  --valid-decoding-path $OUTPUT_PATH \
  --multi-bleu-path $DISK_CODE/scripts/ \
  |& tee $LOG_PATH/train.log
# --keep-interval-updates
# --keep-last-epochs
# --max-epoch
