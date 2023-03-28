import zipfile
import pandas as pd
import sys
sys.path.append("C:\\Projects\\Project1\\Twitter-NLP\\helper_functions\\")
from helper_functions import parse_data_from_file, fit_tokenizer, seq_and_pad, train_val_split
TRAINING_SPLIT = 0.8
NUM_WORDS = 690962
OOV_TOKEN = '<OOV>'
PADDING='post'
EMBEDDING_DIM=16
MAXLEN=120
# Load our data
sentences, labels = parse_data_from_file("C:\\Projects\\Project1\\Twitter-NLP\\data\\dataset\\data.csv")

# Split it into train and val
train_sentences, val_sentences, train_labels, val_labels = train_val_split(sentences, labels, TRAINING_SPLIT)
# print(f"There are {len(train_sentences)} sentences for training.\n")
# print(f"There are {len(train_labels)} labels for training.\n")
# print(f"There are {len(val_sentences)} sentences for validation.\n")
# print(f"There are {len(val_labels)} labels for validation.")

# Fit the tokenizer
tokenizer = fit_tokenizer(train_sentences, NUM_WORDS, OOV_TOKEN)
word_index = tokenizer.word_index

# print(f"Vocabulary contains {len(word_index)} words\n")
# print("<OOV> token included in vocabulary" if "<OOV>" in word_index else "<OOV> token NOT included in vocabulary")

# Turn the text into sequences and pad them to the same length so we can fit it to our model
train_padded_seq = seq_and_pad(train_sentences, tokenizer, PADDING, MAXLEN)
val_padded_seq = seq_and_pad(val_sentences, tokenizer, PADDING, MAXLEN)

# print(f"Padded training sequences have shape: {train_padded_seq.shape}\n")
# print(f"Padded validation sequences have shape: {val_padded_seq.shape}")
