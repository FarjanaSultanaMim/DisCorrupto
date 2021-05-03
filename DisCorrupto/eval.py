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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

import model
import data

import ast


class param_t:
    def __init__(self, tp):
        ntp = []
        
        for k, v in tp:
            try:
                v = ast.literal_eval(v)
                
            except ValueError:
                pass
            except SyntaxError:
                pass
                
            ntp += [("mp_" + k, v) ]
            
        self.__dict__ = dict(ntp)
        
        if "mp_score_type" not in self.__dict__:
            self.__dict__["mp_score_type"] = "Organization" # Defaults to org.
        
        for k, v in ntp:
            self.__setattr__(k, v)


def main(args):
    
    out_dir = args.model_dir
    paramargs = param_t([ln.strip().split("=", 1) for ln in open(os.path.join(out_dir, "param.txt"), "r")])
    
    print("")
    print("Regression evaluator")
    print("  # Score type: {}".format(paramargs.mp_score_type))
    print("  # Output dir: {}".format(out_dir))
    print("  # Param string:\n{}".format(model.param_str(paramargs)))
    print("")

    # Set GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data and Normalize Score
    essayids, essays, org_scores, scores, prompts, scaler = data.load_annotated_essay_with_normalized_score('Path_to_essay_file(.xlsx)', score_source="data/{}Scores.txt".format(paramargs.mp_score_type))

    # Get Persing Sequence (Paragraph Sequence of Persing et. al. 2010)
    pseqs = np.array([data.get_persing_sequence(e, p) for e, p in zip(essays, prompts)])


    # Preprocess essay
    if paramargs.mp_sent_boundary_preprocess or paramargs.mp_only_sent_boundary:
        essays = data.preprocess_essay_longformer_with_sent_boundary(essays, paramargs)
    else:
        essays = data.preprocess_essay_longformer(essays)


    ####################################### Longformer model ##############################

    # Get test set!
    id2idx = dict([(v, k) for k, v in enumerate(essayids)])
    folds = data.load_folds("data/{}Folds.txt".format(paramargs.mp_score_type), id2idx=id2idx)
    
    assert(0 <= args.fold and args.fold <= 4)
    _, _, ts = data.get_fold(folds, args.fold)

    indices = np.arange(len(essays))
    main_essay_t, main_essay_v, score_t, score_v, indices_t, indices_v, prompt_t, prompt_v = [], essays[ts], [], org_scores[ts], [], indices[ts], [], prompts[ts]
    pseq_t, pseq_v = [], pseqs[indices_v]


    # Tokenize all of the essays and map the tokens to thier word IDs.
    tokenizer = model.longformer_tokenizer_roberta()
    input_id_list_v, attention_masks_v = data.longformer_tokenize_and_map_tokens_to_ids(tokenizer, main_essay_v, paramargs)

    
    # Process Presing sequence and create vocabulary

    char2idx = pickle.load(open(os.path.join(out_dir, "char2idx_f{}.pickle".format(args.fold)), "rb"))
    idx2char = pickle.load(open(os.path.join(out_dir, "idx2char_f{}.pickle".format(args.fold)), "rb"))

    seq_dataset_v = data.SeqDataset_longformer(pseq_v, char2idx=char2idx, idx2char=idx2char)
    seq_data_v = seq_dataset_v.vectorized_data
    seq_data_v = data.pad_masks_longformer(seq_data_v)
    seq_data_len_v = seq_dataset_v.vectorized_data_len

    vocab_size_seq = len(char2idx)

    # Combine the training and validation inputs into a TensorDataset.
    score_v = torch.tensor(score_v)
    validation_dataset = TensorDataset(input_id_list_v, attention_masks_v, score_v, seq_data_v, seq_data_len_v)

    # Create the DataLoaders for our training and validation sets.
    batch_size = 2
    validation_dataloader = DataLoader(validation_dataset, sampler = SequentialSampler(validation_dataset), batch_size = batch_size)

    # Create model and print parameters
    main_model = model.create_longformer_enc(paramargs, vocab_size_seq)

    checkpoint = torch.load(os.path.join(args.model_dir, 'checkpoint_f{}.pth'.format(args.fold)))
    main_model.load_state_dict(checkpoint['model_state_dict'])
    main_model = main_model.to(device)



    # Evaluation
    def test(model, iterator, args):
        model.eval()
        prediction_list = []
        ori_score_list = []

        with torch.no_grad():
            for batch in iterator:
                essay, mask, score, seq, seq_len = batch
                essay, mask, seq = essay.to(device), mask.to(device), seq.to(device)
                predictions = model(essay, mask, seq, seq_len).squeeze(1)

                ori_score_list += score.numpy().tolist()
                prediction_list += predictions.cpu().numpy().tolist()
            
        return prediction_list, ori_score_list


    print("Starting evaluation.")
    predictions, ori_scores = test(main_model, validation_dataloader, paramargs)


    # Transform to original score and calculate MSE
    pred = scaler.inverse_transform(np.array(predictions).reshape(1, -1)) 
    pred = pred.ravel().tolist()

    mse, mae = mean_squared_error(ori_scores, pred), mean_absolute_error(ori_scores, pred)

    # Save results to the file.
    with open(os.path.join(out_dir, "prediction_f{}.json".format(args.fold)), "w") as f:
        pr = {
            "system": pred,
            "gold": ori_scores,
            "MSE": mse,
            "MAE": mae,
        }
        json.dump(pr, f)
    
    
    # Print results
    print("MSE: ", mse)
    print("MAE: ", mae)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-fo','--fold', dest='fold', type=int, required=True,
        help="Fold ID ([1, 5]).")
    parser.add_argument(
        '-m','--model-dir', dest='model_dir', type=str, required=True,
        help="Path to model.")

    args = parser.parse_args()
    main(args)



