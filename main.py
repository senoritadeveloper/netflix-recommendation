import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity


data = pd.read_csv("netflixData.csv")
print(data.head())

print(data.isnull().sum())

#print only useful data
data = data[["Title", "Description", "Content Type", "Genres"]]
print(data.head())

#delete null values
data = data.dropna()

#prepare title column because contain symbols

import nltk
import re
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text
data["Title"] = data["Title"].apply(clean)

print(data.Title.head())

#using genre column as the feature to recommend similarities
#cosine similarity refers to the similarity score of 2 vectors, range 0 - 1
#0 means no similarity, 1 means greater similarity

feature = data["Genres"].tolist()
tfidf = text.TfidfVectorizer(input=feature, stop_words="english") #converts raw doc to matrix tf-idf
tfidf_matrix = tfidf.fit_transform(feature)
similarity = cosine_similarity(tfidf_matrix)

indices = pd.Series(data.index,
                    index = data['Title']).drop_duplicates()

def recommendation(title, similarity = similarity):
    index = indices[title]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[0:10]
    movieindices = [i[0] for i in similarity_scores]
    return data['Title'].iloc[movieindices]

print(recommendation("oxygen"))
