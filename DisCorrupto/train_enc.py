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
import re

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, SequentialSampler, WeightedRandomSampler


import model
import data

from pywick.samplers import ImbalancedDatasetSampler


def main(args):

    pstr = model.param_str(args)

    out_dir = hashlib.md5(pstr.encode("utf-8")).hexdigest()
    out_dir = os.path.join("longformer/output_enc", out_dir)
    os.system("mkdir -p {}".format(out_dir))

    with open(os.path.join(out_dir, "param.txt"), "w") as f:
        print(pstr, file=f)

    print("")
    print("Essay Encoder trainer")
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

    #Load essay
    df_icle = data.load_essay_csv('PATH_to_ICLEessays(.csv)')
    if (args.mp_class6 or args.mp_class5to) or (args.mp_class5):
        df_icle = data.prepare_df_for_6class(df_icle, icle=True)
        essay_icle = data.get_essay_array_pretrain(df_icle)
        essay_same_prompt_icle = df_icle['essay_replace_same_prompt'].tolist()
        essay_different_prompt_icle = df_icle['essay_replace_different_prompt'].tolist()
    elif args.mp_test == "select_next_same_prompt":
        original_next_para_icle_t, original_next_para_icle_v, false_next_para_icle_t, false_next_para_icle_v = data.prepare_data_for_nextPara_prediction_same_prompt(df_icle, icle=True)
    else:
        essay_icle = data.get_essay_array_pretrain(df_icle, icle=True)


    df_toefl = data.load_essay_xlsx('PATH_to_TOEFL11essays(.xlsx)')
    if (args.mp_class6 or args.mp_class5to) or (args.mp_class5):
        df_toefl = data.prepare_df_for_6class(df_toefl)
        essay_toefl = data.get_essay_array_pretrain(df_toefl)
        essay_same_prompt_toefl = df_toefl['essay_replace_same_prompt'].tolist()
        essay_different_prompt_toefl = df_toefl['essay_replace_different_prompt'].tolist()
    elif args.mp_test == "select_next_same_prompt":
        original_next_para_toefl_t, original_next_para_toefl_v, false_next_para_toefl_t, false_next_para_toefl_v = data.prepare_data_for_nextPara_prediction_same_prompt(df_toefl)
    else:
        essay_toefl = data.get_essay_array_pretrain(df_toefl)


    df_asap = data.load_essay_csv('/home/acb11171pc/mim/repo/essay_data/ASAP_final.csv')
    essay_asap = data.get_essay_array_pretrain(df_asap)



    # Select which essays to use
    if args.mp_essay_selection == 'ICLEandTOEFL11': 
        if args.mp_test == "select_next_same_prompt":
            original_next_para_t = original_next_para_icle_t + original_next_para_toefl_t
            original_next_para_v = original_next_para_icle_v + original_next_para_toefl_v

            false_next_para_t = false_next_para_icle_t + false_next_para_toefl_t
            false_next_para_v = false_next_para_icle_v + false_next_para_toefl_v
        else:
            essays = np.concatenate((essay_icle, essay_toefl), axis=0)
            if (args.mp_class6 or args.mp_class5to) or (args.mp_class5):
                essay_same_prompt = essay_same_prompt_icle + essay_same_prompt_toefl
                essay_different_prompt = essay_different_prompt_icle + essay_different_prompt_toefl
    elif args.mp_essay_selection == 'AllEssay':
        df_icnale = data.load_essay_xlsx('PATH_to_ICNALEessays(.xlsx)')
        essay_icnale = data.get_essay_array_pretrain(df_icnale)
        essay_icnale = [re.sub('\ufeff', '', e) for e in essay_icnale]

        essays = np.concatenate((essay_icle, essay_toefl, essay_asap, essay_icnale), axis=0)
    
    else:
        if args.mp_test == "select_next_same_prompt":
            original_next_para_t = original_next_para_icle_t
            original_next_para_v = original_next_para_icle_v

            false_next_para_t = false_next_para_icle_t
            false_next_para_v = false_next_para_icle_v
        else:
            essays = essay_icle
            if (args.mp_class6 or args.mp_class5to) or (args.mp_class5):
                essay_same_prompt = essay_same_prompt_icle 
                essay_different_prompt = essay_different_prompt_icle 


    # preprocess essay
    if args.mp_test == "select_next_same_prompt":
        original_next_para_t = original_next_para_t
    else:
        essays = data.preprocess_essay_longformer(essays)
        essays = essays.tolist()
        if (args.mp_class6 or args.mp_class5to) or (args.mp_class5):
            essay_same_prompt = data.preprocess_essay_longformer(essay_same_prompt)
            essay_same_prompt = essay_same_prompt.tolist()
            essay_different_prompt = data.preprocess_essay_longformer(essay_different_prompt)
            essay_different_prompt = essay_different_prompt.tolist()
    

    # Create pre-training samples
    if args.mp_other_pretrain == "nextParaPrediction" and args.mp_test == "select_next_same_prompt":
        main_essay_t, main_essay_v, score_t, score_v = data.create_training_data_next_para_prediction_same_prompt(original_next_para_t, original_next_para_v, false_next_para_t, false_next_para_v)
    elif args.mp_other_pretrain == "nextParaPrediction":
        main_essay_t, main_essay_v, score_t, score_v = data.create_training_data_next_para_prediction(essays, args)
    elif args.mp_other_pretrain == "nextSentPrediction":
        main_essay_t, main_essay_v, score_t, score_v = data.create_training_data_next_sentence_prediction(essays, args)

    elif args.mp_class6:
        main_essay, main_scores = data.create_training_data_6class_essay(essays, essay_same_prompt, essay_different_prompt, args)
    elif args.mp_class4:
        main_essay, main_scores = data.create_training_data_4class_essay(essays, args)
    elif args.mp_class5:
        main_essay, main_scores = data.create_training_data_5class_essay(essays, essay_same_prompt, args)
        print('class5')
    elif args.mp_class5to == 2:
        main_essay, main_scores = data.create_training_data_5class_to2class_essay(essays, essay_same_prompt, args)
        print("\n")
        print('class5 to 2')
    elif args.mp_class5to == 3:
        main_essay, main_scores = data.create_training_data_5class_to3class_essay(essays, essay_same_prompt, args)
        print("\n")
        print('class5 to 3')

    else:
        # Corrupt essays
        if args.mp_shuf == "sentence":
            main_essay, main_scores = data.create_training_data_for_shuffled_essays_longformer(essays, args)
        elif args.mp_shuf == "sentence_moderate":
            main_essay, main_scores = data.create_training_data_for_moderate_shuffled_essays_longformer(essays, args)
        elif args.mp_shuf == "di":
            di_list = data.load_discourse_indicators()
            main_essay, main_scores = data.create_training_data_for_di_shuffled_essays_longformer(essays, di_list, args)
        elif args.mp_shuf == "para":
            essays = [i for i in essays if '\n' in i]
            if args.mp_class3:
                main_essay, main_scores = data.create_training_data_for_moderate_paragraph_shuffled_essays_longformer(essays, args)
            else:
                main_essay, main_scores = data.create_training_data_for_paragraph_shuffled_essays_longformer(essays, args)


    

    ####################################### Pre-training Longformer model ###############################

    # Get training and validation set
    if args.mp_other_pretrain == "nextSentPrediction" or args.mp_other_pretrain == "nextParaPrediction":
        main_essay_t, score_t = data.shuffle_lists(main_essay_t, score_t)
        main_essay_v, score_v = data.shuffle_lists(main_essay_v, score_v)
    else:
        main_essay_t, main_essay_v, score_t, score_v = train_test_split(main_essay, main_scores, test_size=0.2, shuffle=True, random_state=33)


    # Tokenize all of the essays and map the tokens to thier word IDs.
    tokenizer = model.longformer_tokenizer_roberta()
    if args.mp_other_pretrain == "nextParaPrediction" or args.mp_other_pretrain == "nextSentPrediction":
        print("tokenization for other pretraining \n")
        input_id_list_t, attention_masks_t = data.longformer_tokenize_and_map_tokens_to_ids_nextSentPara_pretrain(tokenizer, main_essay_t, args)
        input_id_list_v, attention_masks_v = data.longformer_tokenize_and_map_tokens_to_ids_nextSentPara_pretrain(tokenizer, main_essay_v, args)
    else:
        input_id_list_t, attention_masks_t = data.longformer_tokenize_and_map_tokens_to_ids_pretrain(tokenizer, main_essay_t, args)
        input_id_list_v, attention_masks_v = data.longformer_tokenize_and_map_tokens_to_ids_pretrain(tokenizer, main_essay_v, args)

    # Combine the training and validation inputs into a TensorDataset.
    score_t_dataloader = np.array(score_t)
    score_v_dataloader = np.array(score_v)
    
    score_t = torch.tensor(score_t)
    score_v = torch.tensor(score_v)

    train_dataset = TensorDataset(input_id_list_t, attention_masks_t, score_t)
    validation_dataset = TensorDataset(input_id_list_v, attention_masks_v, score_v)


    # Set batch size
    if args.mp_batch_size == 8:
        batch_size = 8
    elif args.mp_batch_size == 16:
        batch_size = 16
    else:
        if args.mp_preenc != None:
            batch_size = 2
        else:
            batch_size = 4



    # Create the DataLoaders for our training and validation sets.
    if args.mp_class5to:
        # class weighting
        print(" \n Class weighting")
        
        labels_unique_t, counts_t = np.unique(score_t_dataloader, return_counts=True)
        class_weights_t = 1./ torch.tensor(counts_t).float()

        labels_unique_v, counts_v = np.unique(score_v_dataloader, return_counts=True)
        class_weights_v = 1./ torch.tensor(counts_v).float()

        samples_weights_t =  class_weights_t[score_t]
        samples_weights_v =  class_weights_v[score_v]

        sampler_t = WeightedRandomSampler(samples_weights_t, len(score_t), replacement=True)
        sampler_v = WeightedRandomSampler(samples_weights_v, len(score_v), replacement=True)
        
        train_dataloader = DataLoader(train_dataset, sampler = sampler_t, batch_size = batch_size)
        validation_dataloader = DataLoader(validation_dataset, sampler = sampler_v, batch_size = batch_size)
        
        print("Used ImbalancedDatasetSampler \n")

    else:
        train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)
        validation_dataloader = DataLoader(validation_dataset, sampler = SequentialSampler(validation_dataset), batch_size = batch_size)

    


    # Create model and print parameters
    main_model = model.create_enc_pretrain_longformer(args)

    params = list(main_model.named_parameters())
    for p in params:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    # Load pretrained encoder
    if args.mp_preenc != None:
        if args.mp_mixed_pretraining:
            checkpoint = torch.load(os.path.join(args.mp_preenc, 'checkpoint.pth'))
            main_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            checkpoint = torch.load(os.path.join(args.mp_preenc, 'checkpoint.pth'))
            main_model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded Pretrained Model")


    # Instantiate Loss and Optimizer class
    if args.mp_regression:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    learning_rate = args.mp_learning_rate
    optimizer = torch.optim.AdamW(main_model.parameters(), lr=learning_rate)

    # Move model and criterion to GPU
    main_model = main_model.to(device)
    criterion = criterion.to(device)

    # Function for accuracy
    def binary_accuracy(preds, y):
        probs = F.softmax(preds, dim=1)
        winners = probs.argmax(dim=1)
        correct = (winners == y).float() 
        acc = correct.sum() / len(correct)
        return acc

    # Function for training
    def train(model, iterator, optimizer, criterion, args):
        epoch_loss = 0
        accuracy = 0
        model.train()
        for batch in iterator:
            optimizer.zero_grad()

            essay, mask, score = batch
            essay, mask, score = essay.to(device), mask.to(device), score.to(device, dtype= torch.float)
            predictions = model(essay, mask)
            
            loss = criterion(predictions, score)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if not args.mp_regression:
                acc = binary_accuracy(predictions, score)
                accuracy += acc
            
        return epoch_loss / len(iterator)


    # Function for validation
    def evaluate(model, iterator, criterion, args):
        epoch_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for batch in iterator:

                essay, mask, score = batch
                essay, mask, score = essay.to(device), mask.to(device), score.to(device, dtype= torch.float)
                predictions = model(essay, mask)
                
                loss = criterion(predictions, score)
                epoch_loss += loss.item()

                if not args.mp_regression:
                    acc = binary_accuracy(predictions, score)
                    accuracy += acc 

        return epoch_loss / len(iterator)



    # Train the model
    N_EPOCHS = 100
    early_stopping = data.EarlyStoppingPretrainingMSE(out_dir, args, patience=5, verbose=True)
    
    print("Starting training.")
    
    train_loss_list = []
    valid_loss_list = []
    
    for epoch in range(N_EPOCHS):
        
        t0 = time.time()
        
        if not args.mp_regression:
            train_loss, train_acc = train(main_model, train_dataloader, optimizer, criterion, args)
            valid_loss, valid_acc = evaluate(main_model, validation_dataloader, criterion, args)
        else:
            train_loss = train(main_model, train_dataloader, optimizer, criterion, args)
            valid_loss = evaluate(main_model, validation_dataloader, criterion, args)

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        
        print('{} seconds'.format(time.time() - t0))
        if not args.mp_regression:
            print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.5f} | Val. Loss: {valid_loss:.5f}| Train Accuracy: {train_acc:.5f}| Valid Accuracy: {valid_acc:.5f}')
            early_stopping(valid_loss, valid_acc, main_model, optimizer, train_loss, train_acc)
        else:
            print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.5f} | Val. Loss: {valid_loss:.5f}')
            early_stopping(valid_loss, main_model, optimizer, train_loss)
            
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Save training and validation loss
    loss_dict = {"loss_train": train_loss_list, "loss_val": valid_loss_list}

    filename = os.path.join(out_dir, 'loss_log.pickle')
    with open(filename, "wb") as f:
        pickle.dump(loss_dict, f) 


    print()
    print("# Output dir: {}".format(out_dir))
     




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-es','--essay-selection', dest='mp_essay_selection', default="ICLE",
        help="Type of essay data (ICLE, ICLEandTOEFL11, AllEssay).")

    parser.add_argument(
        '-s','--seed', dest='mp_seed', type=int, required=True,
        help="seed number for reproducibility")
    parser.add_argument(
        '-shuf','--shuffle-type', dest='mp_shuf', type=str, 
        help="Shuffling type (sentence, di, para, all).")
    parser.add_argument(
        '-l','--learning-rate', dest='mp_learning_rate', type=float, required=True,
        help="Learning rate e.g., .001") 

    parser.add_argument(
        '-enc','--pretrained-encoder', dest='mp_preenc', type=str,
        help="Path to pretrained encoder.")
    parser.add_argument(
        '-3cls','--3class-classification', dest='mp_class3', action="store_true",
        help="Whether to do a 3 class classification task")
    parser.add_argument(
        '-4cls','--4class-classification', dest='mp_class4', action="store_true",
        help="Whether to do a 4 class classification task")
    parser.add_argument(
        '-6cls','--6class-classification', dest='mp_class6', action="store_true",
        help="Whether to do a 6 class classification task")
    parser.add_argument(
        '-5cls','--5class-classification', dest='mp_class5', action="store_true",
        help="Whether to do a 5 class classification task")
    parser.add_argument(
        '-5cls2class','--5class-to-classification', dest='mp_class5to', type=int,
        help="Whether to do a 5 class converting to 2 class classification task")
    parser.add_argument(
        '-batch','--batch-size', dest='mp_batch_size', type=int,
        help="Specify batch size (by default batch_size=4)")
    parser.add_argument(
        '-sb','--sentence-boundary', dest='mp_sent_boundary', action="store_true",
        help="Whether to insert sentence boundary")
    parser.add_argument(
        '-test','--test-pretrain', dest='mp_test', type=str, 
        help="pre-training testing (first_para, first_sent, all_para_first_sent, select_next_from_other_doc, select_next_same_prompt)")
    parser.add_argument(
        '-mixp','--mixed-pretraining', dest='mp_mixed_pretraining', action="store_true",
        help="Whether to perform several pretraining")

    parser.add_argument(
        '-otp','--other-pretraining', dest='mp_other_pretrain', type=str, 
        help="Other type of pre-training (nextSentPrediction, nextParaPrediction)")
    parser.add_argument(
        '-gat','--global-attention-twice', dest='mp_global_attention_twice', action="store_true",
        help="Whether to perform global attention both on [CLS] and [SEP] token")

    parser.add_argument(
        '-pmt','--pretrained-model-type', dest='mp_pretrained_model_type', type=str, 
        help="Type of pretrained model (Longformer, RoBERTa).")


    args = parser.parse_args()
    main(args)








