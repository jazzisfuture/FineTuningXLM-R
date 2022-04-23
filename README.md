# FineTuningXLM-R
use Fairseq to finetuning XLM-R

## 1. 先决条件
- **从源码安装Fairseq**
- 安装transformers

## 2. 流程
### 简化流程

1. script/tokenize_para.py  处理数据集
2. script/split_file.sh 划分数据集
3. script/gen.sh    二值化
4. 修改config文件内的data和pretraing模型位置
5. script/finetuning.sh
### 2.1 数据集处理
使用huggingface transformers的[tokenizer](https://huggingface.co/docs/tokenizers/python/latest/)进行数据处理
> XLM-R TLM 训练时模型的输入形式为
> ![](https://cdn.nlark.com/yuque/0/2022/png/25459708/1650366181533-579071f4-7428-4469-8830-85ecd3a219fc.png#clientId=ub090d48e-59f6-4&crop=0&crop=0&crop=1&crop=1&from=paste&id=u581f2400&margin=%5Bobject%20Object%5D&originHeight=427&originWidth=1620&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u63aa70b6-e5e7-484b-89c7-f50aa079fe9&title=)


为了对模型进行继续预训练我们要将平行语料处理为：<br />**`<s> a</s></s>b</s>`的形式**<br />Transformers的tokenizer提供了这将两个句子处理为TLM输入的能力
```python
tokenizer.tokenize(the_data,add_special_tokens=True)
```
处理语料的core code
```python
def xlm_tok(data,fout):
    fout = open(fout, 'w', encoding='utf-8')
    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    for line in tqdm(data):
        word_pieces = tok.tokenize(line,add_special_tokens=True)
        new_line = " ".join(word_pieces)
        fout.write('{}\n'.format(new_line))
    fout.close()
```
### 2.2 划分数据集
valid 与 test 各为5k句 剩下的为train<br />脚本来自[facebookresearch/XLM](https://github.com/facebookresearch/XLM)
```shell
pair=zh-en
PARA_PATH=where/is/you/data

# 随机划分数据集
split_data() {
    get_seeded_random() {
        seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
    };
    NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    echo "NLINES: $NLINES"
    NTRAIN=$((NLINES - 10000));
    NVAL=$((NTRAIN + 5000));
    echo $NTRAIN $NVAL
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NTRAIN             > $2;
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NVAL | tail -5000  > $3;
    shuf --random-source=<(get_seeded_random 42) $1 | tail -5000                > $4;
}
split_data $PARA_PATH/$pair.spm.all $PARA_PATH/$pair.train $PARA_PATH/$pair.valid $PARA_PATH/$pair.test
```
### 2.3 二值化数据
使用Fairseq的二值化
```shell
base=/where/is/your/data
fairseq-preprocess \
    --only-source \
    --srcdict $base/xlmr.base/dict.txt \
    --trainpref $base/data/zh-en.train \
    --validpref $base/data/zh-en.valid \
    --testpref $base/data/zh-en.test \
    --destdir data-bin/fintuning \
    --workers 20
```
### 2.4 继续预训练
使用fairseq的继续预训练需要从源码构建fairseq<br />使用pip安装的fairseq无法使用fairseq-hydra-train<br />使用fairseq-hydra-train需要一个config文件<br />**如果提示没有yaml包的话 需要安装pyyaml**
```shell
fairseq-hydra-train -m --config-dir /where/is/your/config/file \
--config-name finetuning_xlmr.yaml #config文件名 
```
```yaml
common:
  fp16: true
  log_format: json
  log_interval: 200

model:
  _name: roberta # 使用robeta的原因时XLMR也是基于RoBERTa
  max_positions: 512
  dropout: 0.1
  attention_dropout: 0.1

checkpoint:
  no_epoch_checkpoints: true
  # 需要继续预训练或微调的模型文件
  restore_file: /home/featurize/finetuningXlmr/xlmr.base/model.pt

task:
  _name: masked_lm
  # 二值化数据后的目录
  data: /home/featurize/finetuningXlmr/data-bin/fintuning
  sample_break_mode: complete
  tokens_per_sample: 512
  
criterion: masked_lm

dataset:
  batch_size: 16
  # max_tokens: 50
  ignore_unused_valid_subsets: true
  skip_invalid_size_inputs_valid_test: true

optimizer:
  _name: adam
  weight_decay: 0.01
  adam_betas: (0.9,0.98)
  adam_eps: 1e-06

lr_scheduler:
  _name: polynomial_decay
  warmup_updates: 10000

optimization:
  clip_norm: 0
  lr: [0.0005]
  max_update: 125000
  update_freq: [16]

```
更多的微调config详见[https://github.com/pytorch/fairseq/tree/main/examples/roberta](https://github.com/pytorch/fairseq/tree/main/examples/roberta)



