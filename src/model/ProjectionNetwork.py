import torch.nn as nn
import torch.nn.functional as F

class ProjectionNetwork(nn.Module):
    """
    A simple projection network that projects an input vector to a specified size using a fully connected layer.
    """
    def __init__(self, input_size, projection_size=100):
        """
        Initializes the projection network.
        :param input_size: Size of the input vector.
        :param projection_size: Size of the projected vector (default is 100).
        """
        super(ProjectionNetwork, self).__init__()
        self.fc = nn.Linear(input_size, projection_size)
        self.activation = nn.ELU()

    def forward(self, x):
        """
        Forward pass of the projection network.
        :param x: Input tensor.
        :return: Projected tensor after applying a linear layer and ELU activation.
        """
        x = self.fc(x)
        x = self.activation(x)
        return x