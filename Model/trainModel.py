import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.model_selection import train_test_split

import random
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')

# load the dataset
news_d = pd.read_csv('.\Model\\train.csv')

column_n = ['id', 'title', 'author', 'text', 'label']
remove_c = ['id','author']
categorical_features = []
target_col = ['label']
text_f = ['title', 'text']

#Import the essential libraries for analysis
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from collections import Counter

ps = PorterStemmer()
wnl = nltk.stem.WordNetLemmatizer()

stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)

# Removed unused clumns
def remove_unused_c(df,column_n=remove_c):

    df = df.drop(column_n,axis=1)
    return df

# Impute null values with None
def null_process(feature_df):

    for col in text_f:
        feature_df.loc[feature_df[col].isnull(), col] = "None"

    return feature_df

def clean_dataset(df):
    # remove unused column
    df = remove_unused_c(df)
    
    #impute null values
    df = null_process(df)

    return df

# Cleaning text from unused characters
def clean_text(text):
    # removing urls
    text = str(text).replace(r'http[\w:/\.]+', ' ')

    # remove everything except characters and punctuation
    text = str(text).replace(r'[^\.\w\s]', ' ')
    text = str(text).replace('[^a-zA-Z]', ' ')
    text = str(text).replace(r'\s\s+', ' ')

    text = text.lower().strip()
    return text

## Nltk Preprocessing include:
# Stop words, Stemming and Lemmatization
# For our project we use only Stop word removal
def nltk_preprocess(text):
    
    text = clean_text(text)
    wordlist = re.sub(r'[^\w\s]','', text).split()
    text = ' '.join([wnl.lemmatize(word) for word in

    wordlist if word not in stopwords_dict])

    return text

# Perform data cleaning on train and test dataset by calling clean_dataset function
df = clean_dataset(news_d)

# apply preprocessing on text through apply method by calling the function nltk_preprocess
df["text"] = df.text.apply(nltk_preprocess)

# apply preprocessing on title through apply method by calling the function nltk_preprocess
df["title"] = df.title.apply(nltk_preprocess)



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf
        tf.random.set_seed(seed)

set_seed(1)

model_name = "bert-base-uncased"

# max sequence length for each document/sentence sample
max_length = 1024

# load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

news_df = news_d[news_d['text'].notna()]
news_df = news_df[news_df["author"].notna()]
news_df = news_df[news_df["title"].notna()]

def prepare_data(df, test_size=0.2, include_title=True, include_author=True):
    texts = []
    labels = []
    
    for i in range(len(df)):
        text = df["text"].iloc[i]
        label = df["label"].iloc[i]
        
        if include_title:
            text = df["title"].iloc[i] + " - " + text
        
        if include_author:
            text = df["author"].iloc[i] + " : " + text
        
        if text and label in [0, 1]:
            texts.append(text)
            labels.append(label)

    return train_test_split(texts, labels, test_size=test_size)

train_texts, valid_texts, train_labels, valid_labels = prepare_data(news_df)

# tokenize the dataset, truncate when passed `max_length`,
# and pad with 0's when less than `max_length`
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)

class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# convert our tokenized data into a torch Dataset
train_dataset = NewsGroupsDataset(train_encodings, train_labels)
valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)

from sklearn.metrics import accuracy_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
    }

training_args = TrainingArguments(
    output_dir='./results', # output directory
    
    num_train_epochs=1, # total number of training epochs
    
    per_device_train_batch_size=10, # batch size per device during training
    
    per_device_eval_batch_size=20, # batch size for evaluation
    
    warmup_steps=100, # number of warmup steps for learning rate scheduler
    
    logging_dir='./logs', # directory for storing logs
    
    load_best_model_at_end=True, # load the best model when finished training (default metric is loss)
    
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=200, # log & save weights each logging_steps
    save_steps=200,
    evaluation_strategy="steps", # evaluate each `logging_steps`
)

# load the model
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

trainer = Trainer(
    model=model, # the instantiated Transformers model to be trained
    args=training_args, # training arguments, defined above
    train_dataset=train_dataset, # training dataset
    eval_dataset=valid_dataset, # evaluation dataset
    compute_metrics=compute_metrics, # the callback that computes metrics of interest
)

# train the model
trainer.train();

# evaluate the current model after training
trainer.evaluate();

# saving the fine tuned model & tokenizer
# model_path = "fake-news-bert-base-uncased"
# model.save_pretrained(model_path)
# tokenizer.save_pretrained(model_path)

# def get_prediction(text, convert_to_label=False):
#     # prepare our text into tokenized sequence
#     inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    
#     # perform inference to our model
#     outputs = model(**inputs)
    
#     # get output probabilities by doing softmax
#     probs = outputs[0].softmax(1)
    
#     # executing argmax function to get the candidate label
#     d = {
#         0: "reliable",
#         1: "fake"
#     }
    
#     if convert_to_label:
#         return d[int(probs.argmax())]
#     else:
#         return int(probs.argmax())