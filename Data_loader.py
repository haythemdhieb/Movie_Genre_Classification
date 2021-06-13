import pandas as pd
import os, sys
import numpy as np
import yaml
from utils import preproces_text, bag_of_words
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing

sys.path.append("./")
all_words=[]
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
#preprocessing text data: cleaning, removing stop words and tokenizing
data=pd.read_csv(config["path_to_data"])
data=data.dropna()
data.reset_index(drop=True,inplace=True)
le=preprocessing.LabelEncoder()
data["label"]=le.fit_transform(data['Genre1'])
number_of_classes=data["label"].nunique()
data["Plot"]=data["Plot"].apply(lambda x:preproces_text(x))
#Get the full vocabulary list
for index, row in enumerate(data):
    all_words.extend(data.loc[index,"Plot"])
#Perform bag of words to get the words embeeding
data["Plot"]=data["Plot"].apply(lambda x:bag_of_words(x,all_words))
X_train = np.array(data["Plot"].tolist(),dtype="float32")
y_train = np.array(data["label"].tolist())
#Convert training data into a pytorch dataset 
class GenreClassficationData(Dataset):
    def __init__(self):
        self.nb_samples = len(X_train)
        self.xdata = X_train
        self.ydata = y_train

    def __getitem__(self, index):
        return (self.xdata[index], self.ydata[index])

    def __len__(self):
        return self.nb_samples

"""
with open("intents.json", "r") as file:
    intents = json.load(file)

all_words = []
tags = []
data = []
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        words_from_sentence = preproces_text(pattern)
        if words_from_sentence != []:
            all_words.extend(words_from_sentence)
            data.append((words_from_sentence, tag))

all_words = mylist = list(dict.fromkeys(all_words))
all_words = sorted(set(all_words))

tags = sorted(set(tags))
# Create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in data:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)
X_train = np.array(X_train, dtype="float32")
y_train = np.array(y_train)
"""
