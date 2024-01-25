import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random
import json
import re

from matplotlib import pyplot as plt
#plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.figsize'] = (4, 2)
plt.rcParams['axes.grid'] = True

from models.transformer_models import GPT

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f'USING DEVICE: {device}')

CHECKPOINT_NAME = 'LM_GPTSmall_Wiki_TR_best'


def load_checkpoint(root_folder, file_name):
    model_full_path = os.path.join(root_folder, file_name+'.pt')
    checkpoint = torch.load(model_full_path, map_location='cpu')
    
    print(f'Checkpoint: {file_name} is loaded successfully')
    
    return checkpoint


# optimizer=None
checkpoint = load_checkpoint('./saved_models', CHECKPOINT_NAME)
hyperparameters = checkpoint['hyperparameters']
saved_time_asctime = checkpoint['saved_time_asctime']
print(f'    Model last saved at: {saved_time_asctime}')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # !!!!!!!!!!!!!!!!!!!!!


seed_everything(hyperparameters['seed'])


# Import tokenizer
tokenizer = Tokenizer.from_file('./turkish_50257_trwiki-20231120-pages-articles_2.json')
tokenizer.post_processor = TemplateProcessing(
    #single='<start> $A <end>',
    single='<start> $A',
    #single='$A',
    #pair="<start> $A <sep> $B:1 <end>:1",
    special_tokens=[
        ('<start>', tokenizer.token_to_id('<start>')),
        ('<end>', tokenizer.token_to_id('<end>')),
    ],
)

encode = lambda s: tokenizer.encode(s).ids # encoder: take a string, output a list of integers
decode = lambda t: tokenizer.decode(t, skip_special_tokens=False) # decoder: take a list of integers, output a string


model = GPT(**hyperparameters['model_config'])
model.to(device)
print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')


def load_model(model, optimizer, root_folder, file_name):
    model_full_path = os.path.join(root_folder, file_name+'.pt')
    checkpoint = torch.load(model_full_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    #optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f'Model: {file_name} is loaded successfully')
    
    return checkpoint

_ = load_model(model, None, './saved_models', CHECKPOINT_NAME)


def generate_beam_search(model, idx, max_new_tokens, max_seq_len, temp=1.0, beam_width=5):

    beams = [{'sequence': idx.clone(), 'score': 0.0}]

    for _ in range(max_new_tokens):
        next_candidates = []

        for beam in beams:
            idx_cond = beam['sequence'][:, -max_seq_len:]

            logits = model(x_input=idx_cond, pad_mask=None)
            logits = logits[:, -1, :]

            probs = F.softmax(logits/temp, dim=-1)

            topk_values, topk_indices = torch.topk(probs, beam_width, dim=-1)

            for i in range(beam_width):
                candidate_sequence = torch.cat([beam['sequence'], topk_indices[:, i].unsqueeze(1)], dim=-1)
                candidate_score = beam['score'] - torch.log(topk_values[:, i]).item()

                next_candidates.append({'sequence': candidate_sequence, 'score': candidate_score})

        beams = sorted(next_candidates, key=lambda x: x['score'], reverse=True)[:beam_width]

    best_sequence = beams[0]['sequence']

    for _b in beams:
        print(f'Beam score: {_b["score"]:.4f}')

    print('\n\n')

    return best_sequence


def generate_multinomial_sampling(model, idx, max_new_tokens, max_seq_len, temp=1.0, topk=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -max_seq_len:]

        logits = model(x_input=idx_cond, pad_mask=None)
        
        logits = logits[:, -1, :]
                   
        if topk is not None:
            _values, _indices = F.softmax(logits/temp, dim=-1).topk(topk, dim=1)
            #probs = F.softmax(logits/temp, dim=-1).topk(topk, dim=-1).values
            probs = _values
        else:
            probs = F.softmax(logits/temp, dim=-1)
        
        # sample from the distribution
        # _idx_next -> shape: [1, num_samples]
        
        _idx_next = torch.multinomial(probs, num_samples=1)
                
        #_idx_next = probs.argmax(dim=-1, keepdim=True) # Full greedy
        idx_next = _indices[:, _idx_next[0]]
        #print(f'idx_next: {idx_next.item()}')
        
        # BREAK EARLY IF <end> TOKEN
        #if _idx_next.item() == tokenizer.token_to_id('<end>'):
        #   break
        
        idx = torch.cat((idx, idx_next), dim=1)
        # return: (1, max_new_tokens (+ input context))
    return idx


def generate_text(model, device, n_tokens, temp, context=None, topk=None, remove_newlines=True):
    """
    Assumes that model is already on "device"
    """
    model.eval()
    
    if context is None:
        #context = torch.zeros((1, 1), dtype=torch.long, device=device)
        context = torch.tensor([encode('')], dtype=torch.long, device=device)
    else:
        context = torch.tensor([encode(context)], dtype=torch.long, device=device)

    generated = decode(
        generate_multinomial_sampling(
            model,
            context, 
            max_new_tokens=n_tokens, 
            max_seq_len=hyperparameters['context_size'], 
            temp=temp,
            topk=topk
        )[0].tolist()
    )
    
    """
    generated = decode(
        generate_beam_search(
            model=model, 
            idx=context, 
            max_new_tokens=n_tokens, 
            max_seq_len=128, 
            temp=temp, 
            beam_width=5
        )[0].tolist()
    )
    """
    
    if remove_newlines:
        generated = generated.replace('\n', '')

    generated_split = generated.split('<end>')
    print(f'[INFO]: total {len(generated_split)} <end> found!')
    
    for idx, generated in enumerate(generated_split):
        print(f'Generated Text [{idx}]: {generated}')
    


# DEFAULTS
_n_tokens = 100
_topk = 150
_temps = [0.70]

while True:
    print('Input query for text generation:')
    print('[Type "q" to quit, "clear" to clear console, "setup" "temps" to change options and temperature]')
    print(f'[Num tokens to generate: {_n_tokens}, Top {_topk} probs]')
    print(f'[Current temperatures: {_temps}]')
    
    query = input('>> ')

    if len(query) < 1:
        continue
    
    if query == 'q':
        break

    if query == 'clear':
        os.system('clear')
        continue

    if query == 'setup':
        query = input('type n_tokens and topk values with comma: ').split(',')
        _n_tokens = int(query[0])
        _topk = int(query[1])
        continue

    if query == 'temps':
        query = input('type temperature values with comma: ').split(',')
        _temps = [float(t) for t in query]
        continue

    for _t in _temps:
        print(f'Prompt: {query}, (Temperatue: {_t}, {_n_tokens} tokens, Top {_topk} probs)')
        #print('Generated Text:')
        generate_text(
            model=model, 
            device=device,
            n_tokens=_n_tokens, 
            temp=_t, 
            context=query,
            topk=_topk,
            remove_newlines=False
        )
        
        print('*'*30)
        print('\n\n')

