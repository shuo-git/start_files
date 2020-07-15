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


TEXT=$DISK_DATA/$DATA

# Preprocess
python3.6 $DISK_CODE/preprocess.py \
  --source-lang en \
  --target-lang de \
  --trainpref $TEXT/valid \
  --validpref $TEXT/test \
  --destdir $DISK_DATA/$DATA/data-bin \
  --workers 32 \
  --srcdict $TEXT/data-bin/dict.en.txt \
  --tgtdict $TEXT/data-bin/dict.de.txt

