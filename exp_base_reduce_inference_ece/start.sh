#!/usr/bin/env bash
export http_proxy="10.222.13.250:32810"
export https_proxy="10.222.13.250:32810"
export PYTHONIOENCODING=UTF-8

DISK1=/apdcephfs/private_vinceswang
DISK2=/apdcephfs/share_916081/vinceswang
DISK_DATA=$DISK1/DATASET
DISK_CODE=$DISK1/code/fairseq-T
SRC=en
TGT=de
SRC_VOCAB=$DISK1/DATASET/wmt14_en_de_stanford/data-bin/dict.$SRC.txt
TGT_VOCAB=$DISK1/DATASET/wmt14_en_de_stanford/data-bin/dict.$TGT.txt

pip list | grep fairseq
if [ $? != 0 ]; then
  echo 'Install Fairseq First'
  cd $DISK_CODE
  pip install --editable .
fi

ALPHA=1.0
SCALE=1.0
WLW=0.1
ER=0.0
SR=0.0
MP=1.0
#for ALPHA in 0.0 0.2 0.4 0.6 0.8 1.0;do
#for SCALE in 2.0 4.0 8.0 16.0 20.0;do
for WLW in 0.0;do
#for ER in 0.5 1.0 2.0;do
#for MP in 0.70 0.75 0.80 0.85 0.90 0.95 1.0;do
DATA=wmt14_en_de_stanford_sampled/10w/split10
DATAORG=wmt14_en_de_stanford
EXP=wmt14-en-de/reduce-inference-ece/on-heldout/exp-re-a$ALPHA-s$SCALE-wlw$WLW-er$ER-sr$SR-mp$MP-nll
CALI=$DISK1/code/Cali-Ana
InfECE=$DISK1/code/InfECE
TER=$DISK1/tools/tercom-0.7.25
RPT=4
#SUFFIX=".rpt$RPT"
SUFFIX=""

CHECKPOINT_DIR=$DISK2/exp/$EXP
mkdir -p $CHECKPOINT_DIR
OUTPUT_PATH=$DISK2/results/$EXP/evaluation
mkdir -p $OUTPUT_PATH
LOG_PATH=$DISK2/results/$EXP/logs
mkdir -p $LOG_PATH
DECODE_PATH=$DISK2/results/$EXP/inference
mkdir -p $DECODE_PATH
TEMP_PATH=$DISK2/results/$EXP/temp
mkdir -p $TEMP_PATH
#cp $DISK_DATA/$DATA/train.* $TEMP_PATH
#cp $DISK_DATA/$DATA/valid.* $TEMP_PATH
#python3 $CALI/repeat_lines.py $TEMP_PATH/train.$SRC $RPT
#python3 $CALI/repeat_lines.py $TEMP_PATH/train.$TGT $RPT

echo 'Prepare valid data'
cp -r $DISK_DATA/$DATA/valid.de $OUTPUT_PATH
cp -r $DISK_DATA/$DATA/test.de $OUTPUT_PATH
sed -i -e 's/@@ //g' $OUTPUT_PATH/valid.de $OUTPUT_PATH/test.de
echo 'Copying pretrained checkpoint'
RESTORE=$DISK2/exp/wmt14-en-de/base-heldout10w/avg_last_10.pt
cp ${RESTORE} $CHECKPOINT_DIR/checkpoint_last.pt

ter(){
    # usage: ter ref hyp
    ref=$1
    hyp=$2
    python ${InfECE}/add_sen_id.py ${ref} ${ref}.ref
    python ${InfECE}/add_sen_id.py ${hyp} ${hyp}.hyp

    java -jar ${TER}/tercom.7.25.jar -r ${ref}.ref -h ${hyp}.hyp -n ${hyp} -s > /dev/null

    python ${InfECE}/parse_xml.py ${hyp}.xml ${hyp}.shifted
    python ${InfECE}/shift_back.py ${hyp}.shifted.text ${hyp}.shifted.label ${hyp}.pra

    rm ${ref}.ref ${hyp}.hyp ${hyp}.ter ${hyp}.sum ${hyp}.sum_nbest \
        ${hyp}.pra_more ${hyp}.pra ${hyp}.xml ${hyp}.shifted.text \
        ${hyp}.shifted.label ${hyp}.shifted.text.sb
    mv ${hyp}.shifted.label.sb ${hyp}.TER
}

gen(){
    # usage: gen checkpoint iteration
    CP=$1
    ITE=$2
    ((SPLIT=ITE%10))
    mkdir -p $TEMP_PATH/ite$ITE
    cp $DISK_DATA/$DATA/valid.* $TEMP_PATH/ite$ITE
    cp $DISK_DATA/$DATA/train.$SPLIT.* $TEMP_PATH/ite$ITE
    CHECKPOINT=$CHECKPOINT_DIR/$CP
    python3 $DISK_CODE/preprocess.py --source-lang $SRC --target-lang $TGT --trainpref $TEMP_PATH/ite$ITE/train.$SPLIT \
        --destdir $TEMP_PATH/ite$ITE/gen-bin --workers 32 --srcdict $SRC_VOCAB --tgtdict $TGT_VOCAB
    mv $TEMP_PATH/ite$ITE/train.$SPLIT$SUFFIX.$TGT $TEMP_PATH/ite$ITE/train.$SPLIT$SUFFIX.ref.$TGT
    for SUBSET in train;do
        echo "Translating $SUBSET.$SPLIT.$SRC @ iteration$ITE"
        GEN=$TEMP_PATH/ite$ITE/$SUBSET.$SPLIT$SUFFIX
        CUDA_VISIBLE_DEVICES=0 python3.6 $DISK_CODE/generate.py \
          $TEMP_PATH/ite$ITE/gen-bin \
          --fp16 \
          -s $SRC \
          -t $TGT \
          --path $CHECKPOINT \
          --gen-subset $SUBSET \
          --lenpen 1.0 \
          --beam 4 \
          --nbest 1 \
          --max-tokens 8192 \
          > $GEN.$TGT.gen
        grep ^H $GEN.$TGT.gen | python3 $CALI/sorted_cut_fairseq_gen.py 2 > $GEN.$TGT
        echo "Labeling $SUBSET.$SPLIT$SUFFIX.$TGT @ iteration$ITE"
        ter $TEMP_PATH/ite$ITE/train.$SPLIT$SUFFIX.ref.$TGT $GEN.$TGT
    done
    echo "Binarizing @ iteration$ITE"
    python3 $DISK_CODE/preprocess.py --source-lang $SRC --target-lang $TGT \
        --trainpref $TEMP_PATH/ite$ITE/train.$SPLIT$SUFFIX --validpref $TEMP_PATH/ite$ITE/valid \
        --destdir $TEMP_PATH/ite$ITE/data-bin --workers 32 --TER-suffix TER --ref-suffix ref \
        --srcdict $SRC_VOCAB --tgtdict $TGT_VOCAB
}

train(){
    # usage: train iteration
    ITE=$1
    k=1
    ((max_epoch=ITE*k))
    if [ "$ITE" = "1" ]
    then
        reset="--reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer"
    else
        reset=" "
    fi
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3.6 $DISK_CODE/train.py $TEMP_PATH/ite$ITE/data-bin \
      -s $SRC -t $TGT \
      --save-dir $CHECKPOINT_DIR \
      --restore-file checkpoint_last.pt ${reset} \
      --load-TER --load-ref \
      --lr 1e-4 --lr-scheduler fixed --force-anneal 1 --lr-shrink 0.9 \
      --weight-decay 0.0 --clip-norm 0.0 --dropout 0.1 \
      --max-sentences 55 \
      --update-freq 50 \
      --arch transformer \
      --optimizer adam --adam-betas '(0.9, 0.98)' \
      --tensorboard-logdir $LOG_PATH \
      --criterion label_smoothed_cross_entropy_inference_ece \
      --label-smoothing 0.0 --ece-alpha $ALPHA --ece-scale $SCALE --forward-ref --wrong-label-weight $WLW --ending-reg-weight $ER --shift-reg-weight $SR --mean-prob $MP \
      --no-progress-bar \
      --log-format simple \
      --log-interval 1 \
      --save-interval-updates 1000 \
      --save-interval 1 \
      --validate-interval 1 \
      --max-epoch ${max_epoch} \
      --beam 4 \
      --remove-bpe \
      --quiet \
      --all-gather-list-size 9022240 \
      --num-ref $DATA=1 \
      --valid-decoding-path $OUTPUT_PATH \
      --multi-bleu-path $DISK_CODE/scripts/ \
      |& tee $LOG_PATH/train$ITE.log
}

infer(){
    for beam in 1 4;do
        for step in {1..10};do
            echo ${step}
            CP=checkpoint${step}.pt
            echo $CP
            CHECKPOINT=$CHECKPOINT_DIR/$CP
            for SUBSET in valid test;do
                GEN=${SUBSET}_${step}.${beam}.gen
                CUDA_VISIBLE_DEVICES=0 python3.6 $DISK_CODE/generate.py \
                  $DISK_DATA/$DATAORG/data-bin \
                  --fp16 \
                  -s $SRC \
                  -t $TGT \
                  --path $CHECKPOINT \
                  --gen-subset $SUBSET \
                  --lenpen 1.0 \
                  --beam ${beam} \
                  --max-sentences 128 \
                  > $DECODE_PATH/$GEN

                sh $DISK_CODE/scripts/compound_split_bleu.sh $DECODE_PATH/$GEN | tee $DECODE_PATH/$GEN.bleu
            done
        done
    done
}

for ITE in {1..10};do
    gen checkpoint_last.pt $ITE
    train $ITE
done

infer


done