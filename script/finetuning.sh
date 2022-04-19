export CUDA_VISIBLE_DEVICES=1
DATA_DIR=/home/yljiang/code/finetuningXlmr/data-bin/fintuning
fairseq-hydra-train -m --config-dir /home/yljiang/code/finetuningXlmr/config \
--config-name finetuning_xlmr.yaml 