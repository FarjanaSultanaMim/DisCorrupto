import bcolz
import pickle
import re
import os
import random
import spacy

import pandas as pd
import numpy as np
import codecs

from nltk import bigrams
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter

from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.parsing.preprocessing import STOPWORDS 
import string

import nltk
from nltk.tokenize import sent_tokenize

#word_tokenizer = nltk.tokenize.TreebankWordTokenizer()
sent_tokenizer = spacy.load('en',disable=['tagger', 'ner'])
word_tokenizer = spacy.load('en',disable=['parser', 'tagger', 'ner'])

MAX_WORDS = 1000


def load_annotated_essay_with_normalized_score(fn_essays, score_source = "./data/OrganizationScores.txt"):
    """
    Getting normalized score and making a dataframe
    """
    
    df_ic = pd.ExcelFile(fn_essays)
    df = df_ic.parse('Sheet1')
    
    # Add scores.
    df_x = pd.read_csv(score_source, delimiter="\t", header=None)
    df_x.columns = "essay_id score".split()
    
    # Special treatment for organization scores (as it contains mutiple scores)
    if "OrganizationScores" in score_source:
        df_x.score = [float(x.split(",")[0]) for x in df_x.score.values]

    def get_score(x):
        q  = df_x[df_x.essay_id == x["Essay Number"]].score.values
        return q[0] if len(q) > 0 else None
        
    df['score'] = df.apply(get_score, axis=1)
    df = df[pd.notna(df.score)]
    
    sc = MinMaxScaler()
    sc.fit(df.score.values.reshape((-1, 1)))
    df['n_score'] = sc.transform(df.score.values.reshape((-1, 1)))

    if "pseq" in df.columns:
        return df["Essay Number"], np.array(df.essay), np.array(df.score), np.array(df.n_score), np.array(df.Prompt), np.array(df.pseq), sc
    else:
        return df["Essay Number"], np.array(df.essay), np.array(df.score), np.array(df.n_score), np.array(df.Prompt), sc


def load_essay_xlsx(path):
    df = pd.ExcelFile(path)
    df = df.parse('Sheet1')
    return df

def load_essay_csv(path):
    df = pd.read_csv(path)
    return df

def load_essay_tsv(path):
    with codecs.open(path, "r", "Shift-JIS", "ignore") as file:
        df = pd.read_table(file, delimiter="\t")
    return df


def get_essay_array_pretrain(dataframe, icle=False):

    essays = np.array(dataframe.essay)

    if icle:
        tokenizer = RegexpTokenizer(r'\w+|\n')
        essays = [e for e in essays if len(tokenizer.tokenize(e)) <= MAX_WORDS]

    return np.array(essays)


def shuffle_lists(essay, score):
    c = list(zip(essay, score))
    random.shuffle(c)

    essay_new, score_new = zip(*c)
    
    return essay_new, score_new

    

def load_folds(fn = "data/OrganizationFolds.txt", id2idx = {}):
    
    return [[id2idx.get(x, x) for x in v.strip().split('\n')] for f, v in re.findall("^Fold ([1-5]):\n([A-Z0-9\n]+)$", open(fn).read(), flags=re.DOTALL|re.MULTILINE)]


def get_fold(folds, i):
    pattern = [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0],
        [2, 3, 4, 0, 1],
        [3, 4, 0, 1, 2],
        [4, 0, 1, 2, 3],
    ]
        
    tr, ts = folds[pattern[i][0]] + folds[pattern[i][1]] + folds[pattern[i][2]] + folds[pattern[i][3]], folds[pattern[i][4]]
    
    random.seed(33)
    random.shuffle(tr)
    trsz = int(len(folds[pattern[i][0]])*3.5)
    
    return (tr[:trsz], tr[trsz:], ts)


def preprocess_essay_longformer(essay_array):

    processed_essay = []
    for e in essay_array:
        
        e = e.lower().strip()
        e = re.sub(r'\t', '', e)
        e = re.sub(r'\n\n', ' \n ', e)
        e = re.sub(r'\n', '\n ', e)
        #e = re.sub(r'\n', '', e)
        
        processed_essay.append(e)

    return np.array(processed_essay, dtype=object)


def preprocess_essay_longformer_with_sent_boundary(essay_array, args):

    processed_essay = []
    for e in essay_array:
        
        e = e.lower().strip()
        e = re.sub(r'\t', '', e)
        e = re.sub(r'\n\n', ' \n ', e)
        e = re.sub(r'\n', '\n ', e)

        if args.mp_only_sent_boundary:
            e = re.sub(r'\n', ' ', e)

            s = [sent.text for sent in sent_tokenizer(e).sents]
            s = [sent.strip() for sent in s]
            org = " \sss ".join(s)
            org = org.strip()
            processed_essay.append(org)
        
        else:
            emp = re.split('\n', e)
            sent_of_each_para = []
            for i in emp:
                s = [sent.text for sent in sent_tokenizer(i).sents]
                s = [sent.strip() for sent in s]
                org = " \sss ".join(s)
                org = org.strip()
                sent_of_each_para.append(org)

            all_essay = ["{}\n".format(i) for i in sent_of_each_para]
            final_essay = ' '.join(all_essay)
            final_essay = re.sub(r'\n   ', '\n  ', final_essay)
            final_essay = re.sub(r'\n  ', '\n ', final_essay)
            final_essay = final_essay.strip()
            processed_essay.append(final_essay)

    return np.array(processed_essay, dtype=object)


def prepare_df_asap(df):
    print(len(df))
    df = df.loc[df['prompt_id']!=7]
    df = df.loc[df['prompt_id']!=8]
    df = df.reset_index(drop=True)
    print(len(df))
    return df



def pad_sequences_longformer(tensor_list, max_len=1200):
    padded = []
    max_len = min(max_len, max(len(tensor) for tensor in tensor_list))
    for tensor in tensor_list:
        ones = torch.ones(max_len).type(torch.long)
        if len(tensor) < max_len:
            ones[:len(tensor)] = tensor
        else:
            ones = tensor[:max_len]
        padded.append(ones)
    return torch.stack(padded)

def pad_masks_longformer(tensor_list, max_len=1200):
    padded = []
    max_len = min(max_len, max(len(tensor) for tensor in tensor_list))
    for tensor in tensor_list:
        zeros = torch.zeros(max_len).type(torch.long)
        if len(tensor) < max_len:
            zeros[:len(tensor)] = tensor
        else:
            zeros = tensor[:max_len]
        padded.append(zeros)
    return torch.stack(padded)


def longformer_tokenize_and_map_tokens_to_ids(tokenizer, essays, args, prompts=None):
    # Tokenize all of the essays and map the tokens to thier word IDs.
    input_id_list = []
    attention_masks = []

    # For every essay...
    for i, essay in enumerate(essays):
        essay = f'{tokenizer.cls_token}{essay}{tokenizer.eos_token}'
        
        input_ids = torch.tensor(tokenizer.encode(essay))
        input_id_list.append(input_ids)
        
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        attention_masks.append(attention_mask)
    
    input_id_list = pad_sequences_longformer(input_id_list)
    attention_masks = pad_masks_longformer(attention_masks)
    attention_masks[:, [0]] = 2     # Attention mask values -- 0: no attention, 1: local attention, 2: global attention

    return input_id_list, attention_masks


def longformer_tokenize_and_map_tokens_to_ids_pretrain(tokenizer, essays, args):
    # Tokenize all of the essays and map the tokens to thier word IDs.
    input_id_list = []
    attention_masks = []

    # For every essay...
    for i, essay in enumerate(essays):
        essay = f'{tokenizer.cls_token}{essay}{tokenizer.eos_token}'
        
        input_ids = torch.tensor(tokenizer.encode(essay))
        input_id_list.append(input_ids)
        
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        attention_masks.append(attention_mask)
    
    input_id_list = pad_sequences_longformer(input_id_list)
    attention_masks = pad_masks_longformer(attention_masks)
    attention_masks[:, [0]] = 2     # Attention mask values -- 0: no attention, 1: local attention, 2: global attention

    return input_id_list, attention_masks




def longformer_tokenize_and_map_tokens_to_ids_nextSentPara_pretrain(tokenizer, pairs, args):
    # Tokenize all of the essays and map the tokens to thier word IDs.
    input_id_list = []
    attention_masks = []

    # For every sent/para pair...
    for i, pair in enumerate(pairs):

        paired_item = f'{tokenizer.cls_token}{pair}{tokenizer.eos_token}'
        
        tokenized_item = tokenizer.encode(paired_item)
        input_ids = torch.tensor(tokenized_item)
        input_id_list.append(input_ids)
        
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

        attention_masks.append(attention_mask)
    
    input_id_list = pad_sequences_longformer(input_id_list)
    attention_masks = pad_masks_longformer(attention_masks)
    attention_masks[:, [0]] = 2     # Attention mask values -- 0: no attention, 1: local attention, 2: global attention

    return input_id_list, attention_masks


 
def get_persing_sequence(essay, prompt):
    
    essay=essay.lower().strip()
    essay_para=re.split('\n',essay)

    transition_words=['also', 'again', 'as well as', 'besides', 'coupled with', 'furthermore', 'in addition', 'likewise', 
                  'moreover', 'similarly', 'accordingly', 'as a result', 'consequently', 'for this reason', 
                  'for this purpose', 'hence', 'otherwise', 'so then', 'subsequently', 'therefore', 'thus', 
                  'thereupon', 'wherefore', 'contrast', 'by the same token', 'conversely', 'instead', 'likewise', 
                  'on one hand', 'on the other hand', 'on the contrary', 'rather', 'similarly', 'yet', 'but', 
                  'however', 'still', 'nevertheless', 'in contrast','here', 'there', 'over there', 'beyond', 'nearly',
                  'opposite', 'under', 'above','to the left', 'to the right', 'in the distance', 'by the way', 
                  'incidentally', 'above all', 'chiefly', 'with attention to', 'especially', 'particularly', 
                  'singularly', 'aside from', 'barring', 'beside', 'except', 'excepting', 'excluding', 'exclusive of',
                  'other than', 'outside of', 'save', 'chiefly', 'especially', 'for instance', 'in particular', 
                  'markedlFy', 'namely', 'particularly', 'including', 'specifically', 'such as', 'as a rule',
                  'as usual', 'for the most part', 'generally', 'generally speaking', 'ordinarily', 'usually', 
                  'or example', 'for instance', 'for one thing', 'as an illustration', 'illustrated with', 
                  'as an example', 'in this case', 'comparatively', 'coupled with', 'correspondingly',
                  'identically', 'likewise', 'similar', 'moreover', 'together with', 'in essence', 
                  'in other words', 'namely', 'that is', 'that is to say', 'in short', 'in brief', 
                  'to put it differently', 'at first', 'first of all', 'to begin with', 'in the first place',
                  'at the same time','for now', 'for the time being', 'the next step', 'in time', 'in turn', 
                  'later on','meanwhile', 'next', 'then', 'soon', 'the meantime', 'later', 'while', 'earlier',
                  'simultaneously', 'afterward', 'in conclusion', 'with this in mind', 'after all', 
                  'all in all', 'all things considered', 'briefly', 'by and large', 'in any case', 'in any event', 
                  'in brief', 'in conclusion', 'on the whole', 'in short', 'in summary', 'in the final analysis', 
                  'in the long run', 'on balance', 'to sum up', 'to summarize', 'finally']

    No_of_label=[]
    score=0


    a1=re.compile(r'\b[Tt]hey\b')
    a2=re.compile(r'\b[Th]hem\b')
    a3=re.compile(r'\b[Mm]y\b')
    a4=re.compile(r'\b[Hh]e\b')
    a5=re.compile(r'\b[Ss]he\b')

    a6=re.compile(r'\b[Aa]gree\b')
    a7=re.compile(r'\b[Dd]isagree\b')
    a8=re.compile(r'\b[Th]ink\b')
    a9=re.compile(r'\b[Oo]pinion\b')

    a10=re.compile(r'\b[Ff]irstly\b')
    a11=re.compile(r'\b[Ss]econdly\b')
    a12=re.compile(r'\b[Tt]hirdly\b')
    a13=re.compile(r'\b[Aa]nother\b')
    a14=re.compile(r'\b[Aa]spect\b')

    a15=re.compile(r'\b[Ss]upport\b')
    a16=re.compile(r'\b[Ii]nstance\b')

    a17=re.compile(r'\b[Cc]onclusion\b')
    a18=re.compile(r'\b[Cc]onclude\b')
    a19=re.compile(r'\b[Tt]herefore\b')
    a20=re.compile(r'\b[[Ss]um\b]')

    a21=re.compile(r'\b[Hh]owever\b')
    a22=re.compile(r'\b[Bb]ut\b')
    a23=re.compile(r'\b[Aa]rgue\b')

    a24=re.compile(r'\b[Ss]olve\b')
    a25=re.compile(r'\b[Ss]olved\b')
    a26=re.compile(r'\b[Ss]olution\b')

    a27=re.compile(r'\b[Ss]hould\b')
    a28=re.compile(r'\b[Ll]et\b')
    a29=re.compile(r'\b[Mm]ust\b')
    a30=re.compile(r'\b[Oo]ught\b')
    

    
    stop_words = set(stopwords.words('english'))

    prompi=RegexpTokenizer(r'\w+').tokenize(prompt.lower())
    promp= [w for w in prompi if not w in stop_words]


    paragraph=0

    sequence=""

    for j in essay_para:

        essay_tokenized = sent_tokenizer(j)
        essay_sent = [sent.text for sent in essay_tokenized.sents]
        paragraph=paragraph+1
        s=0
        No_of_label=[]

        intro=0
        body=0
        rebut=0
        conclude=0

        for i in essay_sent:

            Elaboration = 0
            Prompt = 0
            Transition = 0
            Thesis = 0
            MainIdea = 0
            Support = 0
            Conclusion = 0
            Rebuttal = 0
            Solution = 0
            Suggestion = 0


            s=s+1

            #Elaboration

            b1=a1.findall(i)
            b2=a2.findall(i)
            b3=a3.findall(i)
            b4=a4.findall(i)
            b5=a5.findall(i)

            if len(b1)!=0:
                Elaboration=Elaboration+1
            if len(b2)!=0:
                Elaboration=Elaboration+1
            if len(b3)!=0:
                Elaboration=Elaboration+1
            if len(b4)!=0:
                Elaboration=Elaboration+1
            if len(b5)!=0:
                Elaboration=Elaboration+1 



            if s==1:
                Prompt=Prompt+1
                Thesis=Thesis+1

            if s==len(essay_sent):
                Conclusion=Conclusion+1   

             #Prompt


            content_wordsi=RegexpTokenizer(r'\w+').tokenize(i.lower())  
            content_words=[w for w in content_wordsi if w not in stop_words]

            match_words=[]

            for j in promp:
                if j in content_words:
                    match_words.append(j)

            if len(content_words)!=0:
                Prompt=Prompt+(5/2)*(len(match_words)/len(content_words)) 
            else:
                Prompt=Prompt+(5/2)*0

            #Transition

            word_tokens=word_tokenizer(i)
            word_tokens = [token.text for token in word_tokens]
            #word_tokens=word_tokenizer.tokenize(i)
            if '?' in word_tokens:
                Transition=Transition+1

            n=4
            bi_grami=list(bigrams(RegexpTokenizer(r'\w+').tokenize(i.lower())))
            bi_gram=[' '.join(str(w)for w in l) for l in bi_grami]
            tri_grami=list(bigrams(RegexpTokenizer(r'\w+').tokenize(i.lower())))
            tri_gram=[' '.join(str(w)for w in l) for l in tri_grami]
            four_grami=list(bigrams(RegexpTokenizer(r'\w+').tokenize(i.lower())))
            four_gram=[' '.join(str(w)for w in l) for l in four_grami]

            n_gram=content_wordsi+bi_gram+tri_gram+four_gram

            for k in transition_words:
                if k in n_gram:
                    Transition=Transition+1


            #Thesis

            b6=a6.findall(i)
            b7=a7.findall(i)
            b8=a8.findall(i)
            b9=a9.findall(i)
            
            if len(b6)!=0:
                Thesis=Thesis+1
            if len(b7)!=0:
                Thesis=Thesis+1      
            if len(b8)!=0:
                Thesis=Thesis+1
            if len(b9)!=0:
                Thesis=Thesis+1 
              


            #MainIdea

            b10=a10.findall(i)
            b11=a11.findall(i)
            b12=a12.findall(i)
            b13=a13.findall(i)
            b14=a14.findall(i)
           

            if len(b10)!=0:
                MainIdea=MainIdea+1
            if len(b11)!=0:
                MainIdea=MainIdea+1
            if len(b12)!=0:
                MainIdea=MainIdea+1
            if len(b13)!=0:
                MainIdea=MainIdea+1   
            if len(b14)!=0:
                MainIdea=MainIdea+1
              


            #Support

            b15=a15.findall(i)
            b16=a16.findall(i)

            if len(b15)!=0:
                Support=Support+1
            if len(b16)!=0:
                Support=Support+1

            #Conclusion

            b17=a17.findall(i)
            b18=a18.findall(i)
            b19=a19.findall(i)
            b20=a20.findall(i)

            if len(b17)!=0:
                Conclusion=Conclusion+1
            if len(b18)!=0:
                Conclusion=Conclusion+1
            if len(b19)!=0:
                Conclusion=Conclusion+1
            if len(b20)!=0:
                Conclusion=Conclusion+1
            

            #Rebuttal

            b21=a21.findall(i)
            b22=a22.findall(i)
            b23=a23.findall(i)

            if len(b21)!=0:
                Rebuttal=Rebuttal+1
            if len(b22)!=0:
                Rebuttal=Rebuttal+1
            if len(b23)!=0:
                Rebuttal=Rebuttal+1    

            #Solution

            b24=a24.findall(i)
            b25=a25.findall(i)
            b26=a26.findall(i)

            if len(b24)!=0:
                Solution=Solution+1
            if len(b25)!=0:
                Solution=Solution+1  
            if len(b26)!=0:
                Solution=Solution+1    


            #Suggestion

            b27=a27.findall(i)
            b28=a28.findall(i)
            b29=a29.findall(i)
            b30=a30.findall(i)

            if len(b27)!=0:
                Suggestion=Suggestion+1    
            if len(b28)!=0:
                Suggestion=Suggestion+1
            if len(b29)!=0:
                Suggestion=Suggestion+1    
            if len(b30)!=0:
                Suggestion=Suggestion+1  


            dictn={}
            dictn['Elaboration']=Elaboration.real
            dictn['Transition']=Transition.real
            dictn['Thesis']=Thesis.real
            dictn['MainIdea']=MainIdea.real
            dictn['Support']=Support.real
            dictn['Conclusion']=Conclusion.real
            dictn['Rebuttal']=Rebuttal.real
            dictn['Solution']=Solution.real
            dictn['Suggestion']=Suggestion.real
            dictn['Prompt']=Prompt.real


            s_label=sorted(dictn, key=dictn.get, reverse=True)[:1]

            s_label_s=s_label[0]


            if s_label_s=='Thesis' or s_label_s=='Prompt':
                intro=intro+1
                conclude=conclude+1
            elif s_label_s=='MainIdea' and s<=3:
                body=body+1
                conclude=conclude+1
                rebut=rebut+1
            elif s_label_s=='MainIdea' and s>len(essay_sent)-3:
                body=body+1
                intro=intro+1   
            elif s_label_s=='Elaboration':
                intro=intro+1
                body=body+1
            elif s_label_s=='Support':
                body=body+1
            elif s_label_s=='Suggestion' or s_label_s=='Conclusion':
                body=body+1
                conclude=conclude+1
            elif s_label_s=='Rebuttal' or s_label_s=='solution':
                body=body+1
                rebut=rebut+1  


        if paragraph==1:
            intro=intro+1
        elif paragraph==len(essay_para):
            conclude=conclude+1
        else:
            body=body+1
            rebut=rebut+1

        dict_para={}
        dict_para['I']=intro.real
        dict_para['B']=body.real
        dict_para['C']=conclude.real
        dict_para['R']=rebut.real

        para_label=sorted(dict_para, key=dict_para.get, reverse=True)[:1]

        para_label_s=para_label[0]

        sequence=sequence+para_label_s

    return(sequence)           


def load_discourse_indicators(fn_di = "/home/acb11171pc/mim/repo/DI_list/DI_wo_and.txt"):
    file = open(fn_di)
    data = file.read()
    data = data.splitlines()

    lowered_list = [i.lower() for i in data]
    di_list = [i.split() for i in lowered_list]
    
    return sorted(di_list, key=len, reverse=True)



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, out_dir, args, patience=30, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.out_dir = out_dir
        self.args = args
        
    def __call__(self, val_loss, model, optimizer, train_loss):
        
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, train_loss)
        elif score < self.best_score:
            self.counter+=1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, train_loss)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, optimizer, train_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        torch.save(
        {   'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
            }, os.path.join(self.out_dir, 'checkpoint_f{}.pth'.format(self.args.fold)))
        
        self.val_loss_min = val_loss



class EarlyStoppingPretraining:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, out_dir, args, patience=5, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.out_dir = out_dir
        self.args = args
        
    def __call__(self, val_loss, val_acc, model, optimizer, train_loss, train_acc):
        
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, val_acc, model, optimizer, train_loss, train_acc)
        elif score < self.best_score:
            self.counter+=1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, val_acc, model, optimizer, train_loss, train_acc)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, val_acc, model, optimizer, train_loss, train_acc):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        torch.save(
        {   'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            }, os.path.join(self.out_dir, 'checkpoint.pth'))
        
        self.val_loss_min = val_loss


class EarlyStoppingPretrainingMSE:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, out_dir, args, patience=5, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.out_dir = out_dir
        self.args = args
        
    def __call__(self, val_loss, model, optimizer, train_loss):
        
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, train_loss)
        elif score < self.best_score:
            self.counter+=1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, train_loss)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, optimizer, train_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        torch.save(
        {   'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            }, os.path.join(self.out_dir, 'checkpoint.pth'))
        
        self.val_loss_min = val_loss



# Pretraining

def create_training_data_for_shuffled_essays(refined_essay):
    essay_orig, essay_shf = shuffled_essay(refined_essay)
    total_essay = essay_orig + essay_shf
    
    # The original essays work as positive examples (label: 1), while the shuffled essays work as negative examples (label: 0)
    scores = [1] * len(refined_essay) + [0] * len(refined_essay)
    
    return np.array(total_essay), scores

def create_training_data_for_moderate_shuffled_essays(refined_essay):
    essay_sent = [[sent.text for sent in sent_tokenizer(essay).sents] for essay in refined_essay]
    print(essay_sent[:5])
    index = [i for i,e in enumerate(essay_sent) if len(e)>2]
    print(len(index))
    refined_essay_sent = [essay_sent[i] for i in index]
    print(len(refined_essay_sent))

    essay_orig, essay_shf = sentence_moderate_shuffled_essay(refined_essay_sent)
    total_essay = essay_orig + essay_shf
    
    # The original essays work as positive examples (label: 1), while the shuffled essays work as negative examples (label: 0)
    scores = [1] * len(refined_essay_sent) + [0] * len(refined_essay_sent)
    
    return np.array(total_essay), scores

def create_training_data_for_di_shuffled_essays(refined_essay, di_list):
    essay_orig, essay_shf = di_shuffled_essay(refined_essay, di_list)
    total_essay = essay_orig + essay_shf
    
    # The original essays work as positive examples (label: 1), while the shuffled essays work as negative examples (label: 0)
    scores = [1] * len(refined_essay) + [0] * len(refined_essay)
    
    return np.array(total_essay), scores 

def create_training_data_for_paragraph_shuffled_essays(refined_essay):
    essay_orig, essay_shf = paragraph_shuffled_essay(refined_essay)
    total_essay = essay_orig + essay_shf
    
    # The original essays work as positive examples (label: 1), while the shuffled essays work as negative examples (label: 0)
    scores = [1] * len(refined_essay) + [0] * len(refined_essay)
    
    return np.array(total_essay), scores

def create_training_data_for_moderate_paragraph_shuffled_essays(refined_essay):

    refined_essay = [i for i in refined_essay if (len(re.split('EOP', i)[:-1]))>2]
    print(len(refined_essay))

    essay_orig, essay_shf = paragraph_shuffled_essay(refined_essay)
    essay_shf_2 = paragraph_moderate_shuffled_essay_nea(refined_essay)
    
    total_essay = essay_orig + essay_shf_2 + essay_shf 
    scores = [2] * len(refined_essay) + [1] * len(refined_essay) + [0] * len(refined_essay)
    
    return np.array(total_essay), scores

def create_training_data_6class_essay_nea(refined_essay, essay_same_prompt, essay_different_prompt):

    print(len(refined_essay))

    essay_same_prompt = insert_para_boundary(essay_same_prompt)
    essay_different_prompt = insert_para_boundary(essay_different_prompt)

    essay_orig, essay_shf = paragraph_shuffled_essay(refined_essay)
    essay_shf_2 = paragraph_moderate_shuffled_essay_nea(refined_essay)
    essay_orig_2, drop_essay = drop_essay_nea(refined_essay)
    
    total_essay = essay_orig + essay_shf_2  + drop_essay + essay_same_prompt + essay_different_prompt + essay_shf
    
    scores = [5] * len(refined_essay) + [4] * len(refined_essay) + [3] * len(refined_essay) + [2] * len(refined_essay) + [1] * len(refined_essay) + [0] * len(refined_essay)
    
    return np.array(total_essay), scores

def create_training_data_4class_essay_nea(refined_essay):

    refined_essay = [i for i in refined_essay if (len(re.split('EOP', i)[:-1]))>2]
    print(len(refined_essay))

    essay_orig, essay_shf = paragraph_shuffled_essay(refined_essay)
    essay_shf_2 = paragraph_moderate_shuffled_essay_nea(refined_essay)
    essay_orig_2, drop_essay = drop_essay_nea(refined_essay)
    
    total_essay = essay_orig + essay_shf_2  + drop_essay  + essay_shf
    scores = [3] * len(refined_essay) + [2] * len(refined_essay) + [1] * len(refined_essay) + [0] * len(refined_essay)
    
    return np.array(total_essay), scores

def create_training_data_prompt_shuffle_essay_nea(refined_essay, essay_same_prompt, essay_different_prompt):

    print(len(refined_essay))

    refined_essay = insert_para_boundary(refined_essay)
    essay_same_prompt = insert_para_boundary(essay_same_prompt)
    essay_different_prompt = insert_para_boundary(essay_different_prompt)
    
    total_essay = refined_essay + essay_same_prompt + essay_different_prompt 
    scores = [2] * len(refined_essay) + [1] * len(refined_essay) + [0] * len(refined_essay)
    
    return np.array(total_essay), scores







#### Longformer Shuffling

def shuffled_essay_longformer(essay_list, args):
    shuffle_essay = []
    original_essay = []

    for i in essay_list:
        i = re.sub(r'\n', '',i)

        s = [sent.text for sent in sent_tokenizer(i).sents]

        s = [sent.strip() for sent in s]
        org = " ".join(s)
        org = org.strip()
        original_essay.append(org)

        np.random.shuffle(s)

        nn = " ".join(s)
        nn = nn.strip()
        shuffle_essay.append(nn)

    return original_essay, shuffle_essay


def di_shuffled_essay_longformer(essay_list, di_list):
    shuffle_essay = []
    
    for essay in essay_list:
        # Replace discourse indicators with the predefined list.
        i_di = find_replace_di(essay, di_list)

        # Shuffle the discourse indicators in the essay.
        i_di_shuf = di_change_half(i_di)

        shuffle_essay.append(i_di_shuf)

    return shuffle_essay

def di_change_half(essay):
    
    tks = [tk for tk in essay.split(" ")]
    dis = [tk for tk in tks if tk.startswith("DI_")]
    
    length = len(set(dis))
    half_length = round(len(set(dis))/2)
    
    
    non_shuffle_data = random.sample(set(dis), half_length)
    new_tks = [tk.replace("DI_", "") if tk.startswith("DI_") and tk in non_shuffle_data else tk for tk in tks]
    
    new_tks = [dis.pop()[3:].replace("_", " ") if tk.startswith("DI_") else tk for tk in new_tks]
    new_tks = [tk.replace("_", " ") for tk in new_tks]

    return " ".join(new_tks)



def paragraph_shuffled_essay_longformer(essay_list, args):
    
    shuffle_essay = []
    
    for essay in essay_list:
        emp = re.split('\n', essay)
        
        empt = emp
        np.random.shuffle(empt)
        if args.mp_no_space:
            all_essay = ["{}\n".format(i) for i in empt]
            e = ''.join(all_essay)
        else:
            all_essay = ["{}\n ".format(i) for i in empt]
            e = ''.join(all_essay)
            e = re.sub(r'\n   ', '\n  ', e)
            e = re.sub(r'\n  ', '\n ', e)
        e= e.strip()
        
        shuffle_essay.append(e)
        
    return shuffle_essay


def drop_essay_longformer(essay_list, args):
    dropped_essay = []
    
    for essay in essay_list:
        emp = re.split('\n', essay)
        length = len(set(emp))
        length_drop = round(len(set(emp))*0.3)

        drop_index = random.sample(set(emp), length_drop)
        for i in drop_index:
            emp.remove(i)
        
        if args.mp_no_space:
            all_essay = ["{}\n".format(i) for i in emp]
            e = ''.join(all_essay)
        else:
            all_essay = ["{}\n ".format(i) for i in emp]
            e = ''.join(all_essay)
            e = re.sub(r'\n   ', '\n  ', e)
            e = re.sub(r'\n  ', '\n ', e)
        e= e.strip()
        
        dropped_essay.append(e)
    
    return dropped_essay


def paragraph_moderate_shuffled_essay_longformer(essay_list, args):
    
    shuffle_essay = []

    for essay in essay_list:
        emp = re.split('\n', essay)

        pick1, pick2 = random.sample(emp, 2)
        pick1_i = emp.index(pick1)
        pick2_i = emp.index(pick2)
        
        if pick1_i<pick2_i:
            if pick2_i-pick1_i == 1 and pick1_i!=0:
                copy = emp[pick1_i-1:pick2_i]
                copy.reverse()
                emp[pick1_i-1:pick2_i] = copy
            elif pick2_i-pick1_i == 1 and pick1_i==0:
                copy = emp[:pick2_i+1]
                copy.reverse()
                emp[:pick2_i+1] = copy
            else:
                copy = emp[pick1_i:pick2_i]
                copy.reverse()
                emp[pick1_i:pick2_i] = copy
        else:
            if pick1_i-pick2_i == 1 and pick2_i!=0:
                copy = emp[pick2_i-1:pick1_i]
                copy.reverse()
                emp[pick2_i-1:pick1_i] = copy
            elif pick1_i-pick2_i == 1 and pick2_i==0:
                copy = emp[:pick1_i+1]
                copy.reverse()
                emp[:pick1_i+1] = copy
            else:
                copy = emp[pick2_i:pick1_i]
                copy.reverse()
                emp[pick2_i:pick1_i] = copy

        if args.mp_no_space:
            all_essay = ["{}\n".format(i) for i in emp]
            e = ''.join(all_essay)
        else:
            all_essay = ["{}\n ".format(i) for i in emp]
            e = ''.join(all_essay)
            e = re.sub(r'\n   ', '\n  ', e)
            e = re.sub(r'\n  ', '\n ', e)
        e= e.strip()

        shuffle_essay.append(e)
        
    return shuffle_essay

def sentence_moderate_shuffled_essay_longformer(essay_list):
    
    shuffle_essay = []
    original_essay = []

    for essay in essay_list:
        emp = essay
        emp = [sent.strip() for sent in emp]

        org = " ".join(emp)
        org = org.strip()
        original_essay.append(org)

        pick1, pick2 = random.sample(emp, 2)
        pick1_i = emp.index(pick1)
        pick2_i = emp.index(pick2)
        
        if pick1_i<pick2_i:
            if pick2_i-pick1_i == 1 and pick1_i!=0:
                copy = emp[pick1_i-1:pick2_i]
                copy.reverse()
                emp[pick1_i-1:pick2_i] = copy
            elif pick2_i-pick1_i == 1 and pick1_i==0:
                copy = emp[:pick2_i+1]
                copy.reverse()
                emp[:pick2_i+1] = copy
            else:
                copy = emp[pick1_i:pick2_i]
                copy.reverse()
                emp[pick1_i:pick2_i] = copy
        else:
            if pick1_i-pick2_i == 1 and pick2_i!=0:
                copy = emp[pick2_i-1:pick1_i]
                copy.reverse()
                emp[pick2_i-1:pick1_i] = copy
            elif pick1_i-pick2_i == 1 and pick2_i==0:
                copy = emp[:pick1_i+1]
                copy.reverse()
                emp[:pick1_i+1] = copy
            else:
                copy = emp[pick2_i:pick1_i]
                copy.reverse()
                emp[pick2_i:pick1_i] = copy

        nn = " ".join(emp)
        nn = nn.strip()

        shuffle_essay.append(nn)
        
    return original_essay, shuffle_essay


def new_df_for_modified_essays(df, icle=False):
    
    if icle:
        drop_index_essay = []
        tokenizer = RegexpTokenizer(r'\w+|\n')
        for i in range(df.index[0], len(df)):
            if len(tokenizer.tokenize(df.at[i, 'essay'])) > MAX_WORDS:
                drop_index_essay.append(i)
                    
        df = df.drop(drop_index_essay)
        df = df.reset_index(drop=True)

    if icle==False:
        essays = df['essay'].tolist()
        new_essays = []
        for e in essays:
            e = e.lower().strip()
            e = re.sub(r'\t', '', e)
            e = re.sub(r'\n\n', ' \n', e)
            new_essays.append(e)
        
        df['essay'] = new_essays
    
    drop_index_essay_len = []
    for i in range(df.index[0], len(df)):
        if len(re.split('\n', df.at[i, 'essay']))<=2:
            drop_index_essay_len.append(i)
            
    df = df.drop(drop_index_essay_len)
    df = df.reset_index(drop=True)
    
    prompt_id = df['prompt_id'].tolist()
    prompt_id_essay_len = [len(df.loc[df['prompt_id']==i]) for i in prompt_id]
    delete_prompt = []
    for i,n in enumerate(prompt_id_essay_len):
        if n<=1:
            delete_prompt.append(prompt_id[i])

    drop_index = []
    for dp in delete_prompt:
        for i in range(df.index[0], len(df)):
            if df.at[i, 'prompt_id']==dp:
                drop_index.append(i)
    
    df = df.drop(drop_index)
    df = df.reset_index()

    return df


def replace_essay_segment_from_similar_prompt(df):
    
    replaced_essays = []
    
    for i in range(df.index[0], len(df)):
        essay = df.at[i, 'essay']
        p_id = df.at[i, 'prompt_id']
        
        emp = re.split('\n', essay)
        
        temp = df.loc[df['prompt_id']== p_id]
        essay_numbers = temp.index.values.tolist()
        essay_numbers.remove(i)
        s = random.sample(essay_numbers, 1)[0]
        
        candidate_essay = df.at[s, 'essay']
        candidate_emp = re.split('\n',  candidate_essay)
        pick1, pick2 = random.sample(candidate_emp, 2)
        pick1_i = candidate_emp.index(pick1)
        pick2_i = candidate_emp.index(pick2)
        if pick1_i<pick2_i:
            copy = candidate_emp[pick1_i:pick2_i]
            emp[pick1_i:pick2_i] = copy
        else:
            copy = candidate_emp[pick2_i:pick1_i]
            emp[pick2_i:pick1_i] = copy
            
        
        all_essay = ["{}\n".format(i) for i in emp]
        e = ''.join(all_essay)
        
        replaced_essays.append(e)
        
    return replaced_essays


def replace_essay_segment_from_different_prompt(df):
    
    replaced_essays = []
    
    for i in range(df.index[0], len(df)):
        essay = df.at[i, 'essay']
        p_id = df.at[i, 'prompt_id']
        
        emp = re.split('\n', essay)
        
        temp = df.loc[df['prompt_id']!= p_id]
        essay_numbers = temp.index.values.tolist()
        s = random.sample(essay_numbers, 1)[0]
        
        candidate_essay = df.at[s, 'essay']
        candidate_emp = re.split('\n',  candidate_essay)
        pick1, pick2 = random.sample(candidate_emp, 2)
        pick1_i = candidate_emp.index(pick1)
        pick2_i = candidate_emp.index(pick2)
        if pick1_i<pick2_i:
            copy = candidate_emp[pick1_i:pick2_i]
            emp[pick1_i:pick2_i] = copy
        else:
            copy = candidate_emp[pick2_i:pick1_i]
            emp[pick2_i:pick1_i] = copy
            
        
        all_essay = ["{}\n".format(i) for i in emp]
        e = ''.join(all_essay)
        
        replaced_essays.append(e)
        
    return replaced_essays


def prepare_df_for_6class(df, icle=False):
    if icle:
        new_df = new_df_for_modified_essays(df, icle=True)
    else:
        new_df = new_df_for_modified_essays(df)

    essays_same_prompt = replace_essay_segment_from_similar_prompt(new_df)
    essays_different_prompt = replace_essay_segment_from_different_prompt(new_df)

    new_df['essay_replace_same_prompt'] = essays_same_prompt
    new_df['essay_replace_different_prompt'] = essays_different_prompt

    return new_df


def create_training_data_for_shuffled_essays_longformer(refined_essay, args):
    refined_essay, essay_shf = shuffled_essay_longformer(refined_essay, args)
    total_essay = refined_essay + essay_shf
    
    # The original essays work as positive examples (label: 1), while the shuffled essays work as negative examples (label: 0)
    scores = [1] * len(refined_essay) + [0] * len(refined_essay)
    
    return np.array(total_essay), scores

def create_training_data_for_moderate_shuffled_essays_longformer(refined_essay, args):
    refined_essay = [re.sub(r'\n', '', e) for e in refined_essay]
    essay_sent = [[sent.text for sent in sent_tokenizer(essay).sents] for essay in refined_essay]
    index = [i for i,e in enumerate(essay_sent) if len(e)>2]
    refined_essay = [refined_essay[i] for i in index]
    refined_essay_sent = [essay_sent[i] for i in index]

    refined_essay, essay_shf = sentence_moderate_shuffled_essay_longformer(refined_essay_sent)
    total_essay = refined_essay + essay_shf
    
    # The original essays work as positive examples (label: 1), while the shuffled essays work as negative examples (label: 0)
    scores = [1] * len(refined_essay) + [0] * len(refined_essay)
    
    return np.array(total_essay), scores

def create_training_data_for_di_shuffled_essays_longformer(refined_essay, di_list, args):
    essay_shf = di_shuffled_essay_longformer(refined_essay, di_list)
    total_essay = refined_essay + essay_shf
    
    # The original essays work as positive examples (label: 1), while the shuffled essays work as negative examples (label: 0)
    scores = [1] * len(refined_essay) + [0] * len(refined_essay)
    
    return np.array(total_essay), scores 

def create_training_data_for_paragraph_shuffled_essays_longformer(refined_essay, args):
    essay_shf = paragraph_shuffled_essay_longformer(refined_essay, args)
    total_essay = refined_essay + essay_shf
    
    # The original essays work as positive examples (label: 1), while the shuffled essays work as negative examples (label: 0)
    scores = [1] * len(refined_essay) + [0] * len(refined_essay)
    
    return np.array(total_essay), scores

def create_training_data_for_moderate_paragraph_shuffled_essays_longformer(refined_essay, args):

    refined_essay = [i for i in refined_essay if len(re.split('\n', i))>2]

    essay_shf = paragraph_shuffled_essay_longformer(refined_essay, args)
    essay_shf_2 = paragraph_moderate_shuffled_essay_longformer(refined_essay, args)
    
    total_essay = refined_essay + essay_shf_2 + essay_shf 
    scores = [2] * len(refined_essay) + [1] * len(refined_essay) + [0] * len(refined_essay)

    return np.array(total_essay), scores


def create_training_data_6class_essay(refined_essay, essay_same_prompt, essay_different_prompt, args):

    print(len(refined_essay))

    essay_shf = paragraph_shuffled_essay_longformer(refined_essay, args)
    essay_shf_2 = paragraph_moderate_shuffled_essay_longformer(refined_essay, args)

    drop_essay = drop_essay_longformer(refined_essay, args)
    
    total_essay = refined_essay + essay_shf_2  + drop_essay + essay_same_prompt + essay_different_prompt + essay_shf
    scores = [5] * len(refined_essay) + [4] * len(refined_essay) + [3] * len(refined_essay) + [2] * len(refined_essay) + [1] * len(refined_essay) + [0] * len(refined_essay)
    
    return np.array(total_essay), scores

def create_training_data_5class_essay(refined_essay, essay_same_prompt, args):

    essay_shf = paragraph_shuffled_essay_longformer(refined_essay, args)
    essay_shf_2 = paragraph_moderate_shuffled_essay_longformer(refined_essay, args)

    drop_essay = drop_essay_longformer(refined_essay, args)
    
    total_essay = refined_essay + essay_shf_2  + drop_essay + essay_same_prompt + essay_shf
    scores = [4] * len(refined_essay) + [3] * len(refined_essay) + [2] * len(refined_essay) + [1] * len(refined_essay) + [0] * len(refined_essay) 
    
    return np.array(total_essay), scores


def create_training_data_5class_to2class_essay(refined_essay, essay_same_prompt, args):

    print(len(refined_essay))

    essay_shf = paragraph_shuffled_essay_longformer(refined_essay, args)
    essay_shf_2 = paragraph_moderate_shuffled_essay_longformer(refined_essay, args)

    drop_essay = drop_essay_longformer(refined_essay, args)
    
    total_essay = refined_essay + essay_shf_2  + drop_essay + essay_same_prompt + essay_shf
    scores = [1] * len(refined_essay) + [0] * len(refined_essay) + [0] * len(refined_essay) + [0] * len(refined_essay) + [0] * len(refined_essay) 
    
    return np.array(total_essay), scores

def create_training_data_5class_to3class_essay(refined_essay, essay_same_prompt, args):

    print(len(refined_essay))

    essay_shf = paragraph_shuffled_essay_longformer(refined_essay, args)
    essay_shf_2 = paragraph_moderate_shuffled_essay_longformer(refined_essay, args)

    drop_essay = drop_essay_longformer(refined_essay, args)
    
    total_essay = refined_essay + essay_shf_2  + drop_essay + essay_same_prompt + essay_shf
    scores = [2] * len(refined_essay) + [1] * len(refined_essay) + [1] * len(refined_essay) + [1] * len(refined_essay) + [0] * len(refined_essay) 
    
    return np.array(total_essay), scores


def create_training_data_4class_essay(refined_essay, args):

    refined_essay = [i for i in refined_essay if len(re.split('\n', i))>2]
    print(len(refined_essay))

    essay_shf = paragraph_shuffled_essay_longformer(refined_essay, args)
    essay_shf_2 = paragraph_moderate_shuffled_essay_longformer(refined_essay, args)

    drop_essay = drop_essay_longformer(refined_essay, args)
    
    total_essay = refined_essay + essay_shf_2  + drop_essay  + essay_shf
    scores = [3] * len(refined_essay) + [2] * len(refined_essay) + [1] * len(refined_essay) + [0] * len(refined_essay)
    
    return np.array(total_essay), scores




# other pretraining

def next_sent_prediction_longformer(essay_list, args):

    original_next_sent = []
    false_next_sent = []

    for essay in essay_list:
        emp = essay
        indexes=[]

        half = len(emp)//2
        first_half = emp[:half]
        second_half = emp[half:]

        if args.mp_test == "select_next_from_other_doc":
            other_essay = essay_list[:indx] + essay_list[indx+1:]
            random_essay = random.choice(other_essay)

        for i in range(len(first_half)-1):
            if i in indexes:
                continue

            current_sent = first_half[i]
            next_sent = first_half[i+1]

            indexes.append(i)
            indexes.append(i+1)

            if args.mp_test == "select_next_from_other_doc":
                false_next = random.choice(random_essay)
            else:
                false_next = random.choice(second_half)
                second_half.remove(false_next)

            if not args.mp_sent_boundary:
                pair_with_original_next = current_sent.strip() + " " + next_sent.strip()
                pair_with_false_next = current_sent.strip() + " " + false_next.strip()
            else:
                pair_with_original_next = current_sent.strip() + " \sss " + next_sent.strip()
                pair_with_false_next = current_sent.strip() + " \sss " + false_next.strip()

            original_next_sent.append(pair_with_original_next.strip())
            false_next_sent.append(pair_with_false_next.strip())
    
    return original_next_sent, false_next_sent



def next_para_prediction_longformer(essay_list, args):
    
    original_next_para = []
    false_next_para = []
    
    for essay in essay_list:
        emp = re.split('\n', essay)
        indexes=[]

        half = len(emp)//2
        first_half = emp[:half]
        second_half = emp[half:]

        if args.mp_test == "select_next_from_other_doc":
            other_essay = essay_list[:indx] + essay_list[indx+1:]
            random_essay = random.choice(other_essay)
            random_essay = re.split('\n', random_essay)

        for i in range(len(first_half)-1):
            if i in indexes:
                continue
            
            current_para = first_half[i]
            next_para = first_half[i+1]

            indexes.append(i)
            indexes.append(i+1)

            if args.mp_test == "select_next_from_other_doc":
                false_next = random.choice(random_essay)
            else:
                false_next = random.choice(second_half)
                second_half.remove(false_next)

            pair_with_original_next = current_para.strip() + "\n " + next_para.strip()
            pair_with_original_next = re.sub(r'\n   ', '\n  ', pair_with_original_next)
            pair_with_original_next = re.sub(r'\n  ', '\n ', pair_with_original_next)
            pair_with_original_next = pair_with_original_next.strip()
            
            pair_with_false_next = current_para.strip() + "\n " + false_next.strip()
            pair_with_false_next = re.sub(r'\n   ', '\n  ', pair_with_false_next)
            pair_with_false_next = re.sub(r'\n  ', '\n ', pair_with_false_next)
            pair_with_false_next = pair_with_false_next.strip()

            original_next_para.append(pair_with_original_next)
            false_next_para.append(pair_with_false_next)
        
    return original_next_para, false_next_para





def create_training_data_next_para_prediction(refined_essay, args):

    refined_essay = [i for i in refined_essay if len(re.split('\n', i))>3]
    print("Number of essays with at least 4 paragraphs: ", len(refined_essay))

    essay_t, essay_v = train_test_split(refined_essay, test_size=0.2, shuffle=True, random_state=33)

    original_next_para_t, false_next_para_t = next_para_prediction_longformer(essay_t, args)
    original_next_para_v, false_next_para_v = next_para_prediction_longformer(essay_v, args)

    next_para_t = original_next_para_t + false_next_para_t
    next_para_v = original_next_para_v + false_next_para_v
    
    # The original next para work as positive examples (label: 1), while the false next para work as negative examples (label: 0)

    scores_t = [1] * len(original_next_para_t) + [0] * len(original_next_para_t)
    scores_v = [1] * len(original_next_para_v) + [0] * len(original_next_para_v)
    
    # return np.array(next_para), scores
    return np.array(next_para_t), np.array(next_para_v), scores_t, scores_v



def create_training_data_next_sentence_prediction(refined_essay, args):
    refined_essay = [re.sub(r'\n', '', i) for i in refined_essay]
    essay_sent = [sent_tokenize(essay) for essay in refined_essay]
    
    index = [i for i,e in enumerate(essay_sent) if len(e)>3]
    refined_essay = [refined_essay[i] for i in index]
    refined_essay_sent = [essay_sent[i] for i in index]
    print("Essays that have more than 3 sentence: ", len(refined_essay_sent))

    essay_t, essay_v = train_test_split(refined_essay_sent, test_size=0.2, shuffle=True, random_state=33)

    original_next_sent_t, false_next_sent_t = next_sent_prediction_longformer_new(essay_t, args)
    original_next_sent_v, false_next_sent_v = next_sent_prediction_longformer_new(essay_v, args)

    next_sent_t = original_next_sent_t + false_next_sent_t
    next_sent_v = original_next_sent_v + false_next_sent_v
    

    scores_t = [1] * len(original_next_sent_t) + [0] * len(original_next_sent_t)
    scores_v = [1] * len(original_next_sent_v) + [0] * len(original_next_sent_v)


    # The pair with original next para work as positive examples (label: 1), while the false next para work as negative examples (label: 0)
    
    # return np.array(next_sent), scores
    return np.array(next_sent_t), np.array(next_sent_v), scores_t, scores_v

