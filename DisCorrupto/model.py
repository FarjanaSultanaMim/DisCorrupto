import pandas as pd
import numpy as np

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import RobertaTokenizer, AdamW
from longformer.longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size

import data

def param_str(args):
    keys = [k for k in args.__dict__ if k.startswith("mp_")]
    return "\n".join(["{}={}".format(k[3:], args.__dict__[k]) for k in sorted(keys)])

def MeanOverTime(output):
    return torch.mean(output, dim=1)

class PersingSeq(nn.Module):
    def __init__(self, vocab_size, n_layers, args, bidirectional=True):
        super().__init__()
        self.hidden_dim = args.mp_pseq_encdim
        self.n_layers = n_layers
        
        self.embed, _= create_embedding_layer(vocab_size, args.mp_pseq_embdim, non_trainable=False)
        self.lstm = nn.LSTM(args.mp_pseq_embdim, self.hidden_dim, num_layers=n_layers, bidirectional=bidirectional, 
                            batch_first=True) #no dropuout
        self.dropout = nn.Dropout(args.mp_dropout)
    
    def forward(self, input_seq, lengths):
        embedded = self.embed(input_seq)
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded) #hidden = num_layers * num_directions, batch, hidden_size
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden) 
        
        return hidden



def longformer_tokenizer_roberta():

    config = LongformerConfig.from_pretrained('./Longformer/longformer-base-4096/') 
    config.attention_mode = 'tvm'
    model_for_tokenizer = Longformer.from_pretrained('./Longformer/longformer-base-4096/', config=config)

    # Load the Roberta tokenizer.
    print('Loading RoBERTa tokenizer...')

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
    tokenizer.max_len = model_for_tokenizer.config.max_position_embeddings
    print("Tokenizer max length: ", tokenizer.max_len)

    return tokenizer


class LongformerClassification(nn.Module):
    def __init__(self, hidden_dim, output_dim, longformer):
        super().__init__()
        
        self.longformer = longformer
        self.fc_classification = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, essay, mask):
        longformer_output = self.longformer(essay, attention_mask=mask)
        longformer_output = longformer_output[0]
        output = MeanOverTime(longformer_output)
        output = self.fc_classification(output)
        
        return output


class LongformerClassification_mixed(nn.Module):
    def __init__(self, hidden_dim, output_dim, longformer):
        super().__init__()
        
        self.longformer = longformer
        self.fc_classification_mixed = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, essay, mask):
        longformer_output = self.longformer(essay, attention_mask=mask)
        longformer_output = longformer_output[0]
        output = MeanOverTime(longformer_output)
        output = self.fc_classification_mixed(output)
        
        return output


class MergeLongformerPersingSeq(nn.Module):
    def __init__(self, hidden_dim, output_dim, longformer, seq_model):
        super().__init__()
        
        self.longformer = longformer
        self.seq_model = seq_model
        self.fc_merge = nn.Linear(hidden_dim, output_dim)
        self.out_merge = nn.Sigmoid()
        
    def forward(self, essay, mask, seq, seq_len):
        
        longformer_output = self.longformer(essay, attention_mask=mask)
        longformer_output = longformer_output[0]
        longformer_output = MeanOverTime(longformer_output)
        
        seq_output = self.seq_model(seq, seq_len)
        
        concat_output = torch.cat((longformer_output, seq_output), dim=1)
        output = self.fc_merge(concat_output)
        
        return self.out_merge(output)


def create_longformer_enc(args, vocab_size_seq=None):

    config = LongformerConfig.from_pretrained('./Longformer/longformer-base-4096/') 
    config.attention_mode = 'tvm'
    longformer_model = Longformer.from_pretrained('./Longformer/longformer-base-4096/', config=config)

    output_dim = 1

    hidden_dim = 768 + (args.mp_pseq_encdim*2)
    n_layers = 1
    
    seq_model = PersingSeq(vocab_size_seq, n_layers, args)
    model = MergeLongformerPersingSeq(hidden_dim, output_dim, longformer_model, seq_model)

    return model


def create_enc_pretrain_longformer(args):

    config = LongformerConfig.from_pretrained('./Longformer/longformer-base-4096/') 
    config.attention_mode = 'tvm'
    longformer_model = Longformer.from_pretrained('./Longformer/longformer-base-4096/', config=config)

    
    if args.mp_class3 or args.mp_class5to == 3:
        output_dim = 3
    elif args.mp_class6:
        output_dim = 6
    elif args.mp_class5:
        output_dim = 5
    elif args.mp_class4:
        output_dim = 4
    else:
        output_dim = 2
    
    hidden_dim = 768

    if args.mp_mixed_pretraining:
        model = LongformerClassification_mixed(hidden_dim, output_dim, longformer_model)
    else:
        model = LongformerClassification(hidden_dim, output_dim, longformer_model)

    return model

    