
#Load Dependencies
import argparse, logging

import pickle
import json
import os
import hashlib
import time
import random

import pandas as pd 
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, SequentialSampler

import model
import data

def main(args):

    pstr = model.param_str(args)

    out_dir = hashlib.md5(pstr.encode("utf-8")).hexdigest()
    out_dir = os.path.join("/longformer/output", out_dir)
    os.system("mkdir -p {}".format(out_dir))

    with open(os.path.join(out_dir, "param.txt"), "w") as f:
        print(pstr, file=f)

    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f)

    print("")
    print("Regression trainer")
    print("  # Score type: {}".format(args.mp_score_type))
    print("  # Output dir: {}".format(out_dir))
    print("  # Param string:\n{}".format(pstr))
    print("")


    # Reproducibility
    SEED = args.mp_seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Set GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # Load data and Normalize Score  
    essayids, essays, _, scores, prompts, _ = data.load_annotated_essay_with_normalized_score('Path_to_essay_file(.xlsx)', score_source="data/{}Scores.txt".format(args.mp_score_type))
    
    # Get Persing Sequence (Paragraph Sequence of Persing et. al. 2010)
    pseqs = np.array([data.get_persing_sequence(e, p) for e, p in zip(essays, prompts)])


    # preprocess essay
    if args.mp_sent_boundary_preprocess or args.mp_only_sent_boundary:
        essays = data.preprocess_essay_longformer_with_sent_boundary(essays, args)
    else:
        essays = data.preprocess_essay_longformer(essays)


    ####################################### Longformer model Implementation #############################

    # Get training and validation set
    id2idx = dict([(v, k) for k, v in enumerate(essayids)])
    folds = data.load_folds("./essayScore_folds/{}Folds.txt".format(args.mp_score_type), id2idx=id2idx)

    assert(0 <= args.fold and args.fold <= 4)
    tr, v, ts = data.get_fold(folds, args.fold)

    if args.mp_divide_data == "half":
            _, tr = train_test_split(tr, test_size = 0.50, random_state= args.mp_seed)
    elif args.mp_divide_data == "one_forth":
            _, tr = train_test_split(tr, test_size = 0.25, random_state= args.mp_seed)
    elif args.mp_divide_data == "one_eighth":
            _, tr = train_test_split(tr, test_size = 0.125, random_state= args.mp_seed)  
    else:
        tr= tr
        v=v

    indices = np.arange(len(essays))
    main_essay_t, main_essay_v, score_t, score_v, indices_t, indices_v, prompt_t, prompt_v = essays[tr], essays[v], scores[tr], scores[v], indices[tr], indices[v], prompts[tr], prompts[v]
    pseq_t, pseq_v = pseqs[indices_t], pseqs[indices_v]
    
   
    # Tokenize all of the essays and map the tokens to thier word IDs.
    tokenizer = model.longformer_tokenizer_roberta()
    
    input_id_list_t, attention_masks_t = data.longformer_tokenize_and_map_tokens_to_ids(tokenizer, main_essay_t, args)
    input_id_list_v, attention_masks_v = data.longformer_tokenize_and_map_tokens_to_ids(tokenizer, main_essay_v, args)

    # Process Presing sequence and create vocabulary
    seq_dataset_t = data.SeqDataset_longformer(pseq_t)
    seq_data_t = seq_dataset_t.vectorized_data
    seq_data_t = data.pad_masks_longformer(seq_data_t)
    seq_data_len_t = seq_dataset_t.vectorized_data_len

    with open(os.path.join(out_dir, "char2idx_f{}.pickle".format(args.fold)), "wb") as f:
        pickle.dump(seq_dataset_t.char2idx, f)
    with open(os.path.join(out_dir, "idx2char_f{}.pickle".format(args.fold)), "wb") as f:
        pickle.dump(seq_dataset_t.idx2char, f)

    seq_dataset_v = data.SeqDataset_longformer(pseq_v, char2idx=seq_dataset_t.char2idx, idx2char=seq_dataset_t.idx2char)
    seq_data_v = seq_dataset_v.vectorized_data
    seq_data_v = data.pad_masks_longformer(seq_data_v)
    seq_data_len_v = seq_dataset_v.vectorized_data_len

    vocab_size_seq = len(seq_dataset_t.char2idx)
    

    # Combine the training and validation inputs into a TensorDataset.
    score_t = torch.tensor(score_t)
    score_v = torch.tensor(score_v)
    
    train_dataset = TensorDataset(input_id_list_t, attention_masks_t, score_t, seq_data_t, seq_data_len_t)
    validation_dataset = TensorDataset(input_id_list_v, attention_masks_v, score_v, seq_data_v, seq_data_len_v)

    # set batch size
    batch_size = 2

    # Create the DataLoaders for our training and validation sets.
    train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)
    validation_dataloader = DataLoader(validation_dataset, sampler = SequentialSampler(validation_dataset), batch_size = batch_size)

    # Create model and print parameters
    main_model = model.create_longformer_enc(args, vocab_size_seq)

    # Load pretrained encoder
    if args.mp_preenc != None:
        checkpoint = torch.load(os.path.join(args.mp_preenc, 'checkpoint.pth'))
        main_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("\n Loaded Pretrained Model")


    # Instantiate Loss and Optimizer class
    criterion = nn.MSELoss()
    learning_rate = args.mp_learning_rate

    # Instantiate optimizer
    trainable_parameters = []
    trainable_parameters_name = []
    for name, p in main_model.named_parameters():
        if "longformer" not in name:
            trainable_parameters.append(p)
            trainable_parameters_name.append(name)

    if args.mp_longformer_fix_enc:
        optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
        print(trainable_parameters_name)
    else:
        optimizer = torch.optim.Adam([
            {'params': main_model.longformer.parameters()},
            {'params': trainable_parameters, 'lr': 0.001}
        ], lr=learning_rate)
        print('Merged', trainable_parameters_name)
    

    
    # Move model and criterion to GPU
    main_model = main_model.to(device)
    criterion = criterion.to(device)

    # Function for training
    def train(model, iterator, optimizer, criterion, args):
        epoch_loss = 0
        model.train()
        for batch in iterator:
            
            optimizer.zero_grad()
            
            essay, mask, score, seq, seq_len = batch
            essay, mask, score, seq = essay.to(device), mask.to(device), score.to(device, dtype= torch.float), seq.to(device)
            predictions = model(essay, mask, seq, seq_len).squeeze(1)
            
            loss = criterion(predictions, score)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss / len(iterator)

    # Function for validation
    def evaluate(model, iterator, criterion, args):
        epoch_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in iterator:

                essay, mask, score, seq, seq_len = batch
                essay, mask, score, seq = essay.to(device), mask.to(device), score.to(device, dtype= torch.float), seq.to(device)
                predictions = model(essay, mask, seq, seq_len).squeeze(1)
                
                loss = criterion(predictions, score)
                epoch_loss += loss.item()
            
        return epoch_loss / len(iterator)


    # Train the model
    
    N_EPOCHS = 100
    early_stopping = data.EarlyStopping(out_dir, args, patience=12, verbose=True)
    
    print("Starting training.")
    
    train_loss_list = []
    valid_loss_list = []
    
    for epoch in range(N_EPOCHS):
        
        t0 = time.time()
        
        train_loss  = train(main_model, train_dataloader, optimizer, criterion, args)
        valid_loss = evaluate(main_model, validation_dataloader, criterion, args)
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        
        print('{} seconds'.format(time.time() - t0))
        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.5f} | Val. Loss: {valid_loss:.5f}')
        
        early_stopping(valid_loss, main_model, optimizer, train_loss)
            
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Save training and validation loss
    loss_dict = {"loss_train": train_loss_list, "loss_val": valid_loss_list}

    filename = os.path.join(out_dir, 'loss_log_f{}.pickle'.format(args.fold))
    with open(filename, "wb") as f:
        pickle.dump(loss_dict, f) 

    
    print()
    print("# Output dir: {}".format(out_dir))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-fo','--fold', dest='fold', type=int, required=True,
        help="Fold ID ([1, 5]).")
    parser.add_argument(
        '-sct','--score-type', dest='mp_score_type', default="Organization",
        help="Type of score (Organization, ArgumentStrength, ThesisClarity, PromptAdherence).")
    
    
    # Model parameters.
    parser.add_argument(
         '-d','--dropout', dest='mp_dropout', type=float,
         help="Dropout ratio.")  
    parser.add_argument(
        '-s','--seed', dest='mp_seed', type=int, required=True,
        help="seed number for reproducibility")
    parser.add_argument(
        '-l','--learning-rate', dest='mp_learning_rate', type=float, required=True,
        help="Learning rate e.g., .001")
    # parser.add_argument(
    #     '-gc','--gradientclipnorm', dest='mp_clipnorm', type=float, required=True,
    #     help="Gradient clipping norm.")


    parser.add_argument(
        '-pseq-embdim','--pseq-embedding-dim', dest='mp_pseq_embdim', type=int,
        help="Dimension of PersingNg10 sequence embdding.")
    parser.add_argument(
        '-pseq-encdim','--pseq-encoder-dim', dest='mp_pseq_encdim', type=int,
        help="Dimension of PersingNg10 sequence encoder.")

    parser.add_argument(
        '-enc','--pretrained-encoder', dest='mp_preenc', type=str,
        help="Path to pretrained encoder.")
    parser.add_argument(
        '-dd','--divide-data', dest='mp_divide_data', type=str, 
        help="Type of model (half, one_forth, one_eighth).")
    parser.add_argument(
        '-sbound','--sentence-boundary-preprocess', dest='mp_sent_boundary_preprocess', action="store_true",
        help="Whether to insert sentence boundary")
    parser.add_argument(
        '-onlysbound','--only-sentence-boundary', dest='mp_only_sent_boundary', action="store_true",
        help="Whether to insert sentence boundary")
    parser.add_argument(
        '-pmt','--pretrained-model-type', dest='mp_pretrained_model_type', type=str, 
        help="Type of pretrained model (Longformer, RoBERTa).")
    parser.add_argument(
        '-lfe','--longformer-fix-enc', dest='mp_longformer_fix_enc', action="store_true",
        help="Whether to fix the longformer encoder")
    parser.add_argument(
        '-onlylongformer','--only-longformer', dest='mp_only_longformer', action="store_true",
        help="If Longformer alone should be used")

    args = parser.parse_args()
    main(args)












