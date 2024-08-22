import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# data collection and preprocessing
movie = pd.read_csv("C:/Users/rajes/Downloads/movies.csv")
movie.head()
movie.shape
selected_features = ['genres','keywords','tagline','cast','director']
# print(selected_features)


# replacing the nullm values with null string
for feature in selected_features:
    movie[feature] = movie[feature].fillna('')


# combining all the 5 selected features
combined_features = movie['genres']+' '+ movie['keywords']+' '+ movie['tagline']+' '+ movie['cast']+' '+ movie['director']
# print(combined_features)


# converting text data to feature vector
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)
# print(feature_vector)


# getting similarity scores with cosine similarity
similarity = cosine_similarity(feature_vector)
# print(similarity)
# similarity.shape

# Getting movie name from user
movie_name = input('Enter your favorite movie name: ')

# Creating a list with movie names from the dataset
list_movie = movie['title'].tolist()

# Finding close matches for the movie name given by the user
find_close_matches = difflib.get_close_matches(movie_name, list_movie)

# Iterate over all close matches
for close_match in find_close_matches:
    # Find the index of the movie with the title
    index_of_movie = movie[movie.title == close_match]['index'].values[0]

    # Get a list of similar movies
    similarity_score = list(enumerate(similarity[index_of_movie]))

    # Sort the movies based on similarity score
    sorted_list = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Print the name of similar movies based on the index
    print('Movies suggested for you:')
    i = 0
    for movies in sorted_list:
        index = movies[0]
        title_from_index = movie[movie.index == index]['title'].values[0]
        if i < 5:
            print(i, ',', title_from_index)
        i += 1