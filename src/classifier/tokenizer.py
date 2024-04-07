import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from ProjectionNetwork import ProjectionNetwork

def generate_category_embeddings():
    """
    Generates embeddings for predefined category labels using an embedding layer.
    Returns a dictionary mapping category labels to their embeddings.
    """
    category_labels = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]
    embedding_dim = 30  # Dimension of the embedding vector
    embedding_layer = nn.Embedding(len(category_labels), embedding_dim)

    def label_to_index(label):
        """Converts a category label to its corresponding index."""
        return category_labels.index(label)

    def embed_category(label):
        """Generates an embedding for a given category label."""
        index = label_to_index(label)
        return embedding_layer(torch.LongTensor([index]))

    category_embeddings = {cat: embed_category(cat) for cat in category_labels}
    return category_embeddings

def get_headlines_from_db():
    """
    Placeholder function to fetch headlines and their categories from a database.
    """
    pass

def generate_vectors():
    """
    Generates concatenated vectors of BERT embeddings and category embeddings for headlines.
    Returns a list of projected vectors after processing through a projection network.
    """
    category_embeddings = generate_category_embeddings()
    headlines = get_headlines_from_db()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    projected_vectors = []

    # Process each headline
    for headline, category in headlines.items():
        # Tokenize headline and pass through BERT
        tokens = tokenizer(headline, return_tensors='pt', padding=True, truncation=True)
        # Assuming the model is not fine-tuned and does not require gradients
        with torch.no_grad():
            outputs = bert_model(**tokens)
        # Extract pooled output and concatenate with category embeddings
        pooled_output = outputs.pooler_output
        # concatenate the pooled output with the category embedding
        concatenated_vector = torch.cat((pooled_output, category_embeddings[category]), dim=1)
        projection_network = ProjectionNetwork(798)  # 798 because 768 (BERT output) + 30 (category embedding)
        projected_vector = projection_network(concatenated_vector) 
        projected_vectors.append(projected_vector) # Append the projected vector to the list

    return projected_vectors