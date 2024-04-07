import pickle
import pandas
import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
loaded_tfidf = pickle.load(open("tfidf.pkl","rb"))
loaded_model = pickle.load(open("softmax.pkl","rb"))

category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]

toClassify = ["Meta stock skyrockets 21% after earnings. Here's what Wall Street is saying."]
X_new_counts = loaded_vec.transform(toClassify)
X_new_tfidf = loaded_tfidf.transform(X_new_counts)
predicted = loaded_model.predict(X_new_tfidf)

for i in predicted:
    print(category_list[i])