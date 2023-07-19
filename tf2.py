from transformer import Transformer
import utils

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import torch
import numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = Tokenizer.from_file('tokenizer')
except:
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer()
    tokenizer.train(files=utils.getFiles('transcripts/'), trainer=trainer)
    tokenizer.save('tokenizer', pretty=False)

vocab_size = tokenizer.get_vocab_size()
model = Transformer(vocab_size, vocab_size, 0, 0, device=device)  



tokens = tokenizer.encode("hello world")
x = torch.tensor([tokens.ids]).to(device)

out = model(x, x[:, :-1])
out = out.view(-1, vocab_size)
ids = [i.detach().numpy().argmax()for i in out]
print(tokenizer.decode(ids))





