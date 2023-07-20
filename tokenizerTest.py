from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import numpy as np

import utils

import random

SF = '[<START>]'
EF = '[<END>]'

try:
    tokenizer = Tokenizer.from_file('tokenizer')
except:
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer()
    tokenizer.add_special_tokens([SF, EF, ' '])
    tokenizer.train(files=utils.getFiles('transcripts/'), trainer=trainer)
    tokenizer.save('tokenizer', pretty=False)


def generateData(n):
    SFT = tokenizer.encode(SF).ids[0]
    EFT = tokenizer.encode(EF).ids[0]
    
    f = open("transcripts/EH51mAFpcEQ.txt", "r")
    text = f.read()
    text = text.splitlines()
    text = [SF+t+EF for t in text]
    text = ''.join(text)
    encoded = tokenizer.encode(text)
    encoded_ids = encoded.ids
    MAX_TOKENS_IN = 50
    
    el = len(encoded_ids)-1
    data = []
    i = 0
    while i < n:
        print(f'\tcompling data: {i+1}/{n}\r', end='')
        end = random.randint(0,el)
        start_in = max(0, end-MAX_TOKENS_IN)
        start_out = max(0, end-MAX_TOKENS_IN+1)
        if encoded.tokens[start_in] in [SF, EF, ' ']:
            # print(encoded_ids[max(0, end-MAX_TOKENS):end])
            in_ = encoded_ids[start_in:end]
            in_ = list(np.pad(in_, (MAX_TOKENS_IN-len(in_), 0), 'constant', constant_values=SFT))
            out_ = encoded_ids[start_out:end+1]
            out_ = list(np.pad(out_, (MAX_TOKENS_IN-len(out_), 0), 'constant', constant_values=EFT))
            data.append([in_, out_])
            i += 1
            continue
    print()
    np.random.shuffle(data)
    return data





