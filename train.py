import sys
import yaml
sys.path.append("./")
from Data_loader import GenreClassficationData, all_words,number_of_classes
from model import GenreClassficationModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
with open("./config.yaml","r") as file:
    config= yaml.safe_load(file)
# Set hyperparemeters
hidden_size = config["hidden_size"]
output_size = number_of_classes # number of classes
input_size = len(all_words)
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
num_epochs = config["epoch"]
# loading the data
GenreClassficationData = DataLoader(GenreClassficationData(), batch_size, shuffle=True, num_workers=1)
device = torch.device("cuda")
model = GenreClassficationModel(input_size, hidden_size, output_size).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# begin training
for epoch in range(num_epochs):
    for words, labels in GenreClassficationData:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = loss_function(outputs, labels)
        # backward mouvement
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
    if (epoch + 1) % 100 == 0:
        print(f"epoch {epoch+1}/{num_epochs},loss={loss.item():.4f}")
# save model
print(f"final loss,loss={loss.item():.4f}")
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
}
FILE = "model.pth"
torch.save(data, FILE)

print(f"training complete. file saved to {FILE}")
