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

DATA=wmt14_en_de_stanford_devtest
EXP=wmt14_en_de_stanford_base_reduce_inference_ece-19-4
CALI=$DISK1/code/Cali-Ana
InfECE=$DISK1/code/InfECE
TER=$DISK1/tools/tercom-0.7.25
RPT=1

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
cp $DISK_DATA/$DATA/train.* $TEMP_PATH
cp $DISK_DATA/$DATA/valid.* $TEMP_PATH
python3 $CALI/repeat_lines.py $TEMP_PATH/train.$SRC $RPT
python3 $CALI/repeat_lines.py $TEMP_PATH/train.$TGT $RPT

echo 'Prepare valid data'
cp -r $DISK_DATA/$DATA/valid.de $OUTPUT_PATH
sed -i -e 's/@@ //g' $OUTPUT_PATH/valid.de
echo 'Copying pretrained checkpoint'
RESTORE=$DISK2/exp/wmt14_en_de_stanford_base/avg_last_10.pt
cp ${RESTORE} $CHECKPOINT_DIR/checkpoint_last.pt

ter(){
    # usage: ter ref hyp
    ref=$1
    hyp=$2
    python ${InfECE}/add_sen_id.py ${ref} ${ref}.ref
    python ${InfECE}/add_sen_id.py ${hyp} ${hyp}.hyp

    java -jar ${TER}/tercom.7.25.jar -r ${ref}.ref -h ${hyp}.hyp -n ${hyp} -s

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
    mkdir -p $TEMP_PATH/ite$ITE
    cp $TEMP_PATH/valid.* $TEMP_PATH/ite$ITE
    cp $TEMP_PATH/train.$SRC.rpt$RPT $TEMP_PATH/ite$ITE/train.$SRC
    CHECKPOINT=$CHECKPOINT_DIR/$CP
    for SUBSET in train;do
        echo "Translating $SUBSET.$TGT @ iteration$ITE"
        GEN=$TEMP_PATH/ite$ITE/$SUBSET
        CUDA_VISIBLE_DEVICES=0 python3.6 $DISK_CODE/generate.py \
          $DISK_DATA/$DATA/data-bin \
          --fp16 \
          -s $SRC \
          -t $TGT \
          --path $CHECKPOINT \
          --gen-subset $SUBSET \
          --lenpen 0.6 \
          --beam 4 \
          --nbest $RPT \
          --max-tokens 8192 \
          > $GEN.$TGT.gen
        grep ^H $GEN.$TGT.gen | python3 $CALI/sorted_cut_fairseq_gen.py 2 > $GEN.$TGT
        echo "TER labeling $SUBSET.$TGT @ iteration$ITE"
        ter $TEMP_PATH/train.$TGT.rpt$RPT $GEN.$TGT
    done
    echo "Binarizing @ iteration$ITE"
    python3 $DISK_CODE/preprocess.py --source-lang $SRC --target-lang $TGT \
        --trainpref $TEMP_PATH/ite$ITE/train --validpref $TEMP_PATH/ite$ITE/valid \
        --destdir $TEMP_PATH/ite$ITE/data-bin --workers 32 --TER-suffix TER \
        --srcdict $SRC_VOCAB --tgtdict $TGT_VOCAB
}

train(){
    # usage: train iteration
    ITE=$1
    k=2
    ((max_update=ITE*k))
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
      --load-TER \
      --lr 0.0001 --lr-scheduler fixed --force-anneal 1 --lr-shrink 0.9 \
      --weight-decay 0.0 --clip-norm 0.0 --dropout 0.1 \
      --max-sentences 80 \
      --update-freq 10 \
      --arch transformer \
      --optimizer adam --adam-betas '(0.9, 0.98)' \
      --tensorboard-logdir $LOG_PATH \
      --criterion label_smoothed_cross_entropy_inference_ece \
      --label-smoothing 0.0 --num-bins 20 --ece-scale 20.0 --ece-alpha 0.8 \
      --wrong-token-weight 0.0 \
      --no-progress-bar \
      --log-format simple \
      --log-interval 1 \
      --save-interval-updates 10000 \
      --save-interval 1 \
      --validate-interval 10000 \
      --max-update ${max_update} \
      --beam 1 \
      --remove-bpe \
      --quiet \
      --all-gather-list-size 5222400 \
      --num-ref $DATA=1 \
      --valid-decoding-path $OUTPUT_PATH \
      --multi-bleu-path $DISK_CODE/scripts/ \
      |& tee $LOG_PATH/train$ITE.log
}

infer(){
    for beam in 4;do
        for step in {1..10};do
            echo ${step}
            CP=checkpoint${step}.pt
            CHECKPOINT=$CHECKPOINT_DIR/$CP
            for SUBSET in valid;do
                GEN=${SUBSET}_${step}.${beam}.gen
                echo "Evaluate on $DATA/$SUBSET with $CHECKPOINT"
                CUDA_VISIBLE_DEVICES=0 python3.6 $DISK_CODE/generate.py \
                  $DISK_DATA/$DATA/data-bin \
                  --fp16 \
                  -s $SRC \
                  -t $TGT \
                  --path $CHECKPOINT \
                  --gen-subset $SUBSET \
                  --lenpen 0.6 \
                  --beam ${beam} \
                  --max-sentences 128 \
                  > $DECODE_PATH/$GEN

                sh $DISK_CODE/scripts/compound_split_bleu.sh $DECODE_PATH/$GEN
            done
        done
    done
}

for ITE in {1..5};do
    gen checkpoint_last.pt $ITE
    train $ITE
done

infer


