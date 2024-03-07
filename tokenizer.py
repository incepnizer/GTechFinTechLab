import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from ProjectionNetwork import ProjectionNetwork
def generateEmbeds():
    # Assuming you have a list of category labels
    category_labels = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]

    # Define the size of the embedding dimension
    embedding_dim = 30

    # Create an embedding layer
    embedding_layer = nn.Embedding(len(category_labels), embedding_dim)

    # Define a function to convert category labels to indices
    def label_to_index(label):
        return category_labels.index(label)

    # Define a function to embed category labels
    def embed_category(label):
        index = label_to_index(label)
        return embedding_layer(torch.LongTensor([index]))

    embedDict = {}
    for cat in category_labels:
        embedDict[cat] = embed_category(cat)

    return embedDict

# connect to mongodb and pull the headlines plus category information
def getHeadlines():
    pass
def generateVectors():
    d = generateEmbeds()
    headlines = getHeadlines()
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    ret = []
    for headline, category in headlines.items():
    # Tokenize the headline
        tokens = tokenizer(headline, return_tensors='pt', padding=True, truncation=True)

        # Encode the tokens using BERT
        with torch.no_grad():
            outputs = model(**tokens)

        # Get the pooled output (CLS token)
        pooled_output = outputs.pooler_output

        concatVector = torch.cat((pooled_output, d[category]), dim=1)

        projNet = ProjectionNetwork(798)
        projVector = projNet.forward(concatVector)
        ret.append(projVector)
    return ret

