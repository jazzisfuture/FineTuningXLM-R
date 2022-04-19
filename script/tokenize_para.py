from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import argparse
MAX_LENGTH=512

parser = argparse.ArgumentParser(description='Tokenize para')
parser.add_argument('--data-a', type=str, required=True, help='one of the parallel data')
parser.add_argument('--data-b', type=str, required=True, help='another one of the parallel data')
parser.add_argument('--save-path', type=str, required=True, help='save path')
args = parser.parse_args()

if not args.data_a or not args.data_b:
    print('Please input the data path')
    exit()

def xml_tok(data,fout):
    fout = open(fout, 'w', encoding='utf-8')
    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    for line in tqdm(data):
        word_pieces = tok.tokenize(line,add_special_tokens=True)
        new_line = " ".join(word_pieces)
        fout.write('{}\n'.format(new_line))
    fout.close()

with open(args.data_a, 'r', encoding='utf-8') as f:
    data_a = [ line.replace(" ","").replace('\n','') for line in f.readlines()]
with open(args.data_b, 'r', encoding='utf-8') as f:
    data_b = [ line.replace('\n','') for line in f.readlines()]
assert len(data_a) == len(data_b)
data = list(zip(data_a,data_b))

xml_tok(data,args.save_path+'/processed.spm.all')

print("+++++++++done+++++++++")

