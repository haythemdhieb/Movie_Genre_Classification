import torch.nn as nn
import sys

sys.path.append("./")
# Model creation
class GenreClassficationModel(nn.Module):
    """
    The architecture of  model that will be used for classification
    """

    def __init__(self, input_size, hidden_size, num_classes):
        super(GenreClassficationModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.layer1(x)
        output = self.relu(output)
        output = self.layer2(output)
        output = self.relu(output)
        output = self.layer3(output)
        return output
