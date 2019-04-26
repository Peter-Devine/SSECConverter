import argparse
import os
import pandas as pd
import numpy as np

# Takes input and output directories as arguments
parser=argparse.ArgumentParser()
parser.add_argument('--input', default=".", help='The file path of the unzipped Grounded Emotions dataset')
parser.add_argument('--output', default="./data", help='The file path of the output dataset')
parser.add_argument('--sensitivity', default="33", help='Select sensitivity of emotion detection (0, 33, 5, 66 or 99)')

args = parser.parse_args()
INPUT_PATH = args.input
OUTPUT_PATH = args.output
SENSITIVITY = str(int(args.sensitivity))

if SENSITIVITY not in ["0", "33", "5", "66", "99"]:
    SENSITIVITY = "33"

# Make the output directory if it does not currently exist
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

emotions = ["Anger", "Anticipation", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Trust"]

def train_dev_split(train_df):
    fraction = 0.25

    np.random.seed(seed=42)

    dev_indices = np.random.choice(train_df.index, size=int(round(fraction*train_df.shape[0])), replace=False)
    train_indices = train_df.index.difference(dev_indices)

    train = train_df.loc[train_indices,:]
    dev = train_df.loc[dev_indices,:]
    
    return(train, dev)

for database_type in ["train", "test"]:
    
    ssec = pd.read_csv(INPUT_PATH + "/"+database_type+"-combined-0."+SENSITIVITY+".csv", sep="\t", names= emotions + ["Text"])

    for emotion in emotions:
        mask = ssec[emotion] != "---"
        ssec[emotion.lower() + "_present"] = mask.astype(int)
        
    if database_type == "train":
        ssec_train, ssec_dev = train_dev_split(ssec)
        
        ssec_train.reset_index(drop=True).to_csv(OUTPUT_PATH+"/train.tsv", sep='\t', encoding="utf-8")
        ssec_dev.reset_index(drop=True).to_csv(OUTPUT_PATH+"/dev.tsv", sep='\t', encoding="utf-8")
    else:
        ssec.to_csv(OUTPUT_PATH+"/test.tsv", sep='\t', encoding="utf-8")