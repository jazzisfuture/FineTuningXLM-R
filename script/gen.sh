base=/home/yljiang/code/finetuningXlmr
fairseq-preprocess \
    --only-source \
    --srcdict $base/xlmr.base/dict.txt \
    --trainpref $base/data/zh-en.train \
    --validpref $base/data/zh-en.valid \
    --testpref $base/data/zh-en.test \
    --destdir data-bin/fintuning \
    --workers 20