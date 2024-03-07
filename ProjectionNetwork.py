import torch.nn as nn
import torch.nn.functional as F

# Define the size of the projected vector
projection_size = 100

# Define the fully connected feedforward neural network
class ProjectionNetwork(nn.Module):
    def __init__(self, input_size):
        super(ProjectionNetwork, self).__init__()
        self.fc = nn.Linear(input_size, projection_size)  # Fully connected layer
        self.activation = nn.ELU()  # ELU activation function

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x