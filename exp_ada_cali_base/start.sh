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
EXP=${DATA}_ada_cali_base
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


CUDA_VISIBLE_DEVICES=0,1,2,3 python3.6 $DISK_CODE/train.py $DISK_DATA/$DATA/data-bin \
  --fp16 \
  -s en -t de \
  --num-bins 20 --moving-alpha 0.5 --err-alpha 20 \
  --lr 0.0007 --min-lr 1e-09 \
  --weight-decay 0.0 --clip-norm 0.0 --dropout 0.1 \
  --max-tokens 8192 \
  --update-freq 1 \
  --arch transformer \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-07 \
  --warmup-updates 4000 \
  --save-dir $CHECKPOINT_DIR \
  --tensorboard-logdir $LOG_PATH \
  --criterion label_smoothed_cross_entropy_moving_avg \
  --label-smoothing 0.1 \
  --no-progress-bar \
  --log-format simple \
  --log-interval 100 \
  --save-interval-updates 2000 \
  --max-update 100000 \
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
