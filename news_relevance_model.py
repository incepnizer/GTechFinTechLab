import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from ProjectionNetwork import ProjectionNetwork

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Compute scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.input_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

class NewsRelevanceModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', category_embedding_dim=30, projection_dim=100, num_classes=3):
        super(NewsRelevanceModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.category_embedding = nn.Embedding(num_embeddings=7, embedding_dim=category_embedding_dim)  # Assuming 7 categories
        self.projection = ProjectionNetwork(input_size=768 + category_embedding_dim)  # BERT output + category embedding
        self.attention = Attention(input_dim=projection_dim)
        self.classifier = nn.Linear(projection_dim, num_classes)
    
    def forward(self, input_ids, attention_mask, category_indices):
        '''
        input_ids: these are tokenized indices from bert tokenizer (news headlines tokenized via Bert). Shape is [batch_size, max_seq_len]

        attention_mask: this is a mask that tells the model where the actual tokens are and where the padding is. The attention mask has the same shape as input_ids ([batch_size, sequence_length]). 

        category_indices: integers (0-6) representing the categories of news articles, used to fetch embeddings that enhance the BERT output with categorical context. The shape is [batch_size], with each value indicating a news article's category.

        '''
        # Process headlines through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, 768]
        
        # Embed categories
        category_embeddings = self.category_embedding(category_indices)  # [batch_size, category_embedding_dim]
        
        # Concatenate BERT output and category embeddings
        concatenated = torch.cat((pooled_output, category_embeddings), dim=1)  # [batch_size, 768 + category_embedding_dim]
        
        # Project concatenated vectors
        projected = self.projection(concatenated)  # [batch_size, projection_dim]
        
        # Apply attention
        attention_output, _ = self.attention(projected.unsqueeze(1))  # [batch_size, 1, projection_dim]
        attention_output = attention_output.squeeze(1)  # [batch_size, projection_dim]
        
        # Classify
        logits = self.classifier(attention_output)  # [batch_size, num_classes]
        
        return logits

# Example usage
# model = NewsRelevanceModel()
# we need to prepare input_ids, attention_mask, and category_indices based on dataset 
# logits = model(input_ids, attention_mask, category_indices)