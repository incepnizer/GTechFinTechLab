from datasets import load_dataset
import pickle
import pandas
import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pymongo

def getData():
    dataset = load_dataset("gigaword")
    return dataset

def train(docArr):
    loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
    loaded_tfidf = pickle.load(open("tfidf.pkl","rb"))
    loaded_model = pickle.load(open("softmax.pkl","rb"))

    X_new_counts = loaded_vec.transform(docArr)
    X_new_tfidf = loaded_tfidf.transform(X_new_counts)
    predicted = loaded_model.predict(X_new_tfidf)
    return predicted

def main():
    dataset = getData()
    smalld = dataset['train']
    docArr = []
    for thing in smalld:
        docArr.append(thing['document'])

    docPlusScore = zip(docArr, train(docArr))
    category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]
    data = {}
    for doc, score in docPlusScore:
        data[doc] = category_list[score]
    
    updateDb(data)

def updateDb(data):
    # Connection details
    mongo_host = 'localhost'  # MongoDB host
    mongo_port = 27017  # MongoDB port
    mongo_db = 'VIP2'  # Name of the database
    collection_name = 'FinTechReplication'  # Name of the collection

    # Connect to MongoDB
    client = pymongo.MongoClient(mongo_host, mongo_port)

    # Access the database
    db = client[mongo_db]

    # Access the collection
    collection = db[collection_name]

    for sentence, classification in data.items():
        dataInsert = {
            "sentence": sentence,
            "classification": classification
        }
        collection.insert_one(dataInsert)
    print("Data added successfully.")


if __name__ == '__main__':
    main()