from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import utils

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

e=tokenizer.encode(f'{SF}hello world{EF}')
print(e.ids)





