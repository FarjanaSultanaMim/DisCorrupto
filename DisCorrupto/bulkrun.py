import sys
import os


comm_train_longformer = ["""

# 1: Base + AE
python src/train.py \
    --fold {} \
    --score-type {} \
    --dropout 0.7 --seed 5444 --learning-rate 1e-5 \
    --pretrained-model-type longformer \
    --pseq-embedding-dim 16 --pseq-encoder-dim 200
""",

"""
# 2: Base + AE (fix-encoder)
python src/train.py \
    --fold {} \
    --score-type {} \
    --dropout 0.7 --seed 5444 --learning-rate 0.001 \
    --pretrained-model-type longformer --longformer-fix-enc \
    --pseq-embedding-dim 16 --pseq-encoder-dim 200
""",




"""
# 3: Base + AE (5-way DC pre-training, fine_tuned)
python src/train.py \
    --fold {} \
    --score-type {} \
    --dropout 0.7 --seed 5444 --learning-rate 1e-5 \
    --pretrained-model-type longformer \
    --pseq-embedding-dim 16 --pseq-encoder-dim 200 \
    --pretrained-encoder  PATH_TO_PRETRAINED_ENCODER
""",

"""
# 4: Base + AE (5-way DC pre-training, fixed-encoder)
python src/train.py \
    --fold {} \
    --score-type {} \
    --dropout 0.7 --seed 5444 --learning-rate 0.001 \
    --pretrained-model-type longformer --longformer-fix-enc \
    --pseq-embedding-dim 16 --pseq-encoder-dim 200 \
    --pretrained-encoder PATH_TO_PRETRAINED_ENCODER
""",

]





comm_train_enc_longformer = [

"""
# 1: Pre-Pretraining with paragraph-based shuffling
python src/train_enc.py \
    --pretrained-model-type longformer \
    --essay-selection ICLEandTOEFL11 \
    --seed 5999 --learning-rate 1e-5 \
    --shuffle-type para 
""",

"""
# 2: Pretraining with paragraph-based shuffling
python src/train_enc.py \
    --pretrained-model-type longformer \
    --essay-selection icle \
    --seed 8777 --learning-rate 1e-5 \
    --shuffle-type para \
    --pretrained-encoder PATH_TO_PRETRAINED_ENCODER
""",



"""
# 3: Pre-Pretraining with sentence-based shuffling
python src/train_enc.py \
    --pretrained-model-type longformer \
    --essay-selection AllEssay \
    --seed 5999 --learning-rate 1e-5 \
    --shuffle-type sentence
""",

"""
# 4: Pretraining with sentence-based shuffling
python src/train_enc.py \
    --pretrained-model-type longformer \
    --essay-selection icle \
    --seed 8777 --learning-rate 1e-5 \
    --shuffle-type sentence \
    --pretrained-encoder PATH_TO_PRETRAINED_ENCODER
""",



"""
# 5: Pre-Pretraining with DI-based shuffling
python src/train_enc.py \
    --pretrained-model-type longformer \
    --essay-selection AllEssay \
    --seed 5999 --learning-rate 1e-5 \
    --shuffle-type di
""",
"""
# 6: Pretraining with di-based shuffling
python src/train_enc.py \
    --pretrained-model-type longformer \
    --essay-selection icle \
    --seed 8777 --learning-rate 1e-5 \
    --shuffle-type di \
    --pretrained-encoder PATH_TO_PRETRAINED_ENCODER
""",





"""
# 7: Pre-Pretraining with paragraph-based shuffling (3-class)
python src/train_enc.py \
    --pretrained-model-type longformer \
    --essay-selection ICLEandTOEFL11 \
    --seed 5999 --learning-rate 1e-5 \
    --shuffle-type para --3class-classification
""",
"""
# 8: Pretraining with paragraph-based shuffling (3-class)
python src/train_enc.py \
    --pretrained-model-type longformer \
    --essay-selection icle \
    --seed 8777 --learning-rate 1e-5 \
    --shuffle-type para --3class-classification\
    --pretrained-encoder PATH_TO_PRETRAINED_ENCODER
""",



"""
# 9: Pre-Pretraining with paragraph-based shuffling (4-class)
python src/train_enc.py \
    --pretrained-model-type longformer \
    --essay-selection ICLEandTOEFL11 \
    --seed 5999 --learning-rate 1e-5 \
    --shuffle-type para --4class-classification
""",
"""
# 10: Pretraining with paragraph-based shuffling (4-class)
python src/train_enc.py \
    --pretrained-model-type longformer \
    --essay-selection icle \
    --seed 8777 --learning-rate 1e-5 \
    --shuffle-type para --4class-classification\
    --pretrained-encoder PATH_TO_PRETRAINED_ENCODER
""",


"""
# 11: Pre-Pretraining with paragraph-based shuffling (5-class)
python src/train_enc.py \
    --pretrained-model-type longformer \
    --essay-selection ICLEandTOEFL11 \
    --seed 5999 --learning-rate 1e-5 \
    --shuffle-type para --5class-classification
""",
"""
# 12: Pretraining with paragraph-based shuffling (5-class)
python src/train_enc.py \
    --pretrained-model-type longformer \
    --essay-selection icle \
    --seed 8777 --learning-rate 1e-5 \
    --shuffle-type para --5class-classification\
    --pretrained-encoder PATH_TO_PRETRAINED_ENCODER
""",




"""
# 13: Pre-Pretraining with 5-class to 2 class
python src/train_enc.py \
    --pretrained-model-type longformer \
    --essay-selection ICLEandTOEFL11 \
    --seed 5999 --learning-rate 1e-5 \
    --5class-to-classification 2
""",
"""
# 14: Pretraining with 5-class to 2 class
python src/train_enc.py \
    --pretrained-model-type longformer \
    --essay-selection icle \
    --seed 8777 --learning-rate 1e-5 \
    --5class-to-classification 2 \
    --pretrained-encoder PATH_TO_PRETRAINED_ENCODER
""",






#other pretraining


"""
# 15: Pre-Pretraining with next para prediction
python src/train_enc.py \
    --pretrained-model-type longformer \
    --essay-selection ICLEandTOEFL11 \
    --seed 5999 --learning-rate 1e-5 \
    --other-pretraining nextParaPrediction \
""",
"""
# 16: Pretraining with next para prediction
python src/train_enc.py \
    --pretrained-model-type longformer \
    --essay-selection icle \
    --seed 8777 --learning-rate 1e-5 \
    --other-pretraining nextParaPrediction \
    --pretrained-encoder PATH_TO_PRETRAINED_ENCODER
""",


#next sentence prediction

"""
# 17: Pre-Pretraining with next sentence prediction
python src/train_enc.py \
    --pretrained-model-type longformer \
    --essay-selection AllEssay \
    --seed 5999 --learning-rate 2e-5 \
    --other-pretraining nextSentPrediction \
""",
"""
# 18: Pretraining with next sentence prediction
python src/train_enc.py \
    --pretrained-model-type longformer \
    --essay-selection icle \
    --seed 8777 --learning-rate 1e-5 \
    --other-pretraining nextSentPrediction \
    --pretrained-encoder PATH_TO_PRETRAINED_ENCODER
""",

]



comm_eval = ["""
# eval
python src/eval.py \
    --fold {} \
    --model-dir output/
""",
]

comm_eval_homo = """
# EVAL
python src/eval.py \
    --fold {} \
    --model-dir {}
"""



if sys.argv[1] == "train_allfolds":
    sct = sys.argv[2]
    f = int(sys.argv[3])
    comm = [comm_train_longformer[f].format(i, sct) for i in range(0, 5)]


elif sys.argv[1] == "train_allfolds_for_hptune":
    sct = sys.argv[2]
    f = int(sys.argv[3])
    comm = []
    for i in range(0, 5):
        for dropout in [0.5, 0.7, 0.9]:
            cmd = comm_train_longformer[f]
            cmd = cmd.replace("--dropout 0.7", "--dropout {}".format(dropout))
            comm += [cmd.format(i, sct)]
            
            
elif sys.argv[1] == "eval_allfolds_homo":
    model_dir = sys.argv[2]
    comm = [comm_eval_homo.format(i, model_dir) for i in range(0, 5)]
    

elif sys.argv[1] == "eval_allfolds_after_hptune":
    f = int(sys.argv[2])
    comm = []  
    folder = [  
             ]
    
    for i in range(0, 5):
        cmd = comm_eval[f]
        if i==0:
            cmd = cmd.replace("--model-dir output/", "--model-dir {}".format(folder[0]))
        if i==1:
            cmd = cmd.replace("--model-dir output/", "--model-dir {}".format(folder[1]))
        if i==2:
            cmd = cmd.replace("--model-dir output/", "--model-dir {}".format(folder[2]))
        if i==3:
            cmd = cmd.replace("--model-dir output/", "--model-dir {}".format(folder[3]))
        if i==4:
            cmd = cmd.replace("--model-dir output/", "--model-dir {}".format(folder[4]))
        comm += [cmd.format(i)]

for c in comm:
    print("===")
    print("bulkrun.py:", c)
    
    os.system(c)
