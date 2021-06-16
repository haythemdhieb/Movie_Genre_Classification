# Module imports
import pandas as pd
import os, sys
import numpy as np
import yaml
from utils import preproces_text, bag_of_words
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing

sys.path.append("./")
all_words = []
with open("src/config.yaml", "r") as file:
    config = yaml.safe_load(file)
# Preprocessing text data: cleaning, removing stop words and tokenizing
data = pd.read_csv(config["path_to_data"])
data = data.dropna()
data.reset_index(drop=True, inplace=True)
le = preprocessing.LabelEncoder()
data["label"] = le.fit_transform(data["Genre1"])
list_of_classes = dict(zip(data["label"].unique(), data["Genre1"].unique()))
number_of_classes = data["label"].nunique()
data["Plot"] = data["Plot"].apply(lambda x: preproces_text(x))
# Get the full vocabulary list
for index, row in enumerate(data):
    all_words.extend(data.loc[index, "Plot"])
# Perform bag of words to get the words embeeding
data["Plot"] = data["Plot"].apply(lambda x: bag_of_words(x, all_words))
X_train = np.array(data["Plot"].tolist(), dtype="float32")
y_train = np.array(data["label"].tolist())
# Convert training data into a pytorch dataset
class GenreClassficationData(Dataset):
    def __init__(self):
        self.nb_samples = len(X_train)
        self.xdata = X_train
        self.ydata = y_train

    def __getitem__(self, index):
        return (self.xdata[index], self.ydata[index])

    def __len__(self):
        return self.nb_samples
