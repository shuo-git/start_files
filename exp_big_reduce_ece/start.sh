export http_proxy="10.222.13.250:32810"
export https_proxy="10.222.13.250:32810"
export PYTHONIOENCODING=UTF-8

DISK1=/apdcephfs/private_vinceswang
DISK_DATA=$DISK1/DATASET
DISK_CODE=$DISK1/code/fairseq-T
DATA=wmt14_en_de_stanford_devtest

pip list | grep fairseq
if [ $? != 0 ]; then
  echo 'Install Fairseq First'
  cd $DISK_CODE
  pip install --editable .
fi

DISK2=/apdcephfs/share_916081/vinceswang
EXP=wmt14_en_de_stanford_big_reduce_ece-10
CHECKPOINT_DIR=$DISK2/exp/$EXP
mkdir -p $CHECKPOINT_DIR

OUTPUT_PATH=$DISK2/results/$EXP/evaluation
LOG_PATH=$DISK2/results/$EXP/logs

mkdir -p $OUTPUT_PATH
mkdir -p $LOG_PATH

echo 'Prepare valid data'
cp -r $DISK_DATA/$DATA/valid.de $OUTPUT_PATH
sed -i -e 's/@@ //g' $OUTPUT_PATH/valid.de

RESTORE=$DISK2/exp/wmt14_en_de_stanford_big/avg_last_10.pt
cp ${RESTORE} $CHECKPOINT_DIR/checkpoint_last.pt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3.6 $DISK_CODE/train.py $DISK_DATA/$DATA/data-bin \
  -s en -t de \
  --save-dir $CHECKPOINT_DIR \
  --restore-file checkpoint_last.pt \
  --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \
  --lr 0.0001 --lr-scheduler fixed --force-anneal 1 --lr-shrink 0.9 \
  --weight-decay 0.0 --clip-norm 0.0 \
  --max-sentences 80 \
  --update-freq 10 \
  --arch transformer_vaswani_wmt_en_de_big \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --tensorboard-logdir $LOG_PATH \
  --criterion label_smoothed_cross_entropy_train_ece \
  --label-smoothing 0.1 --num-bins 20 --ece-scale 10.0 --ece-alpha 0.0 \
  --no-progress-bar \
  --log-format simple \
  --log-interval 1 \
  --save-interval-updates 100 \
  --save-interval 1 \
  --validate-interval 1 \
  --max-update 10 \
  --beam 1 \
  --remove-bpe \
  --quiet \
  --all-gather-list-size 5222400 \
  --num-ref $DATA=1 \
  --valid-decoding-path $OUTPUT_PATH \
  --multi-bleu-path $DISK_CODE/scripts/ \
  |& tee $LOG_PATH/train.log
# --keep-interval-updates
# --keep-last-epochs
# --max-epoch
