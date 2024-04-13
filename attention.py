from lib2to3.pgen2.tokenize import tokenize
import torch
import torch.nn as nn
import torch.nn.functional as F
import tokenizer

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.W = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input_tensors):
        scores = torch.cat([self.v(torch.tanh(self.W(tensor))).unsqueeze(0) for tensor in input_tensors], dim=0)
        
        weights = F.softmax(scores, dim=0)
    
        weighted_tensors = torch.stack([torch.matmul(weights[i].T, input_tensors[i].unsqueeze(0)) for i in range(len(input_tensors))], dim=0)
        aggregated_tensor = torch.sum(weighted_tensors, dim=0)
        
        print(aggregated_tensor)
        output = self.fc(aggregated_tensor)
        
        return output

input_dim = 100
hidden_dim = 100
output_dim = 3
num_headlines = 10 

attention_layer = AttentionLayer(input_dim, hidden_dim, output_dim)
headline_vectors = tokenizer.generateVectors()

final_prediction = attention_layer(headline_vectors)
print(f" final prediction: {final_prediction}")
predicted_probabilities = torch.softmax(final_prediction, dim=2)
print(f" predicted probabilities: {predicted_probabilities}")