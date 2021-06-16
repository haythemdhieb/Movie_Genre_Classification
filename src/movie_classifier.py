import random
import json, sys
import torch
import argparse
import numpy as np
import yaml
import os

sys.path.append("./")
from model import GenreClassficationModel
from utils import bag_of_words, preproces_text

with open("src/config.yaml", "r") as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser(description="movie title and description")
parser.add_argument("--title", type=str, required=True, help="the movie title")
parser.add_argument("--description", type=str, required=True, help="description of the movie")
args = parser.parse_args()
MovieDescription = args.description
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE = config["File"]
model = torch.load(FILE)
# Retrieve the model's hyperparameters
input_size = model["input_size"]
hidden_size = model["hidden_size"]
output_size = model["output_size"]
all_words = model["all_words"]
model_state = model["model_state"]
classes = model["classes"]
# Load trained model's parameters
model = GenreClassficationModel(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
# We generate the prediction
X = preproces_text(MovieDescription)
X = bag_of_words(X, all_words)
X = X.reshape(1, X.shape[0])
X = torch.from_numpy(X).to(device)
output = model(X)
_, predicted = torch.max(output, dim=1)
probs = torch.softmax(output, dim=1)
response = {
    "title": args.title,
    "description": MovieDescription,
    "genre": classes[predicted.tolist()[0]],
}
print(response)
