#This script will contain the logic 
#for our content-based recommender system.


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, movie_data_path):
        self.movie_data = pd.read_csv(movie_data_path)
        self.tfidf_matrix = self._create_tfidf_matrix()
        self.similarity_matrix = self._calculate_similarity_matrix()

    def _create_tfidf_matrix(self):
        tfidf = TfidfVectorizer(stop_words='english')
        return tfidf.fit_transform(self.movie_data['overview'].fillna(''))

    def _calculate_similarity_matrix(self):
        return cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def recommend_movies(self, movie_title, num_recommendations=5):
        movie_indices = pd.Series(self.movie_data.index, index=self.movie_data['title']).drop_duplicates()
        idx = movie_indices.get(movie_title, None)

        if idx is None:
            return []

        similarity_scores = list(enumerate(self.similarity_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:num_recommendations + 1]

        movie_indices = [i[0] for i in similarity_scores]
        return self.movie_data['title'].iloc[movie_indices].tolist()
