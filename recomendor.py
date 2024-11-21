import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, movie_data_path):
        """
        Initialize the ContentBasedRecommender class.
        Load the movie data, compute the TF-IDF matrix, and calculate the similarity matrix.
        
        Args:
            movie_data_path (str): Path to the CSV file containing movie data.
        """
        # Load movie data from a CSV file
        self.movie_data = pd.read_csv(movie_data_path)
        # Create the TF-IDF matrix from the movie overviews
        self.tfidf_matrix = self._create_tfidf_matrix()
        # Compute the similarity matrix based on the TF-IDF vectors
        self.similarity_matrix = self._calculate_similarity_matrix()

    def _create_tfidf_matrix(self):
        """
        Create a TF-IDF matrix based on the 'overview' column of the movie data.
        
        Returns:
            scipy.sparse.csr_matrix: TF-IDF matrix where each row represents a movie.
        """
        # Initialize a TF-IDF vectorizer with English stop words
        tfidf = TfidfVectorizer(stop_words='english')
        # Fit and transform the 'overview' column, replacing missing values with an empty string
        return tfidf.fit_transform(self.movie_data['overview'].fillna(''))

    def _calculate_similarity_matrix(self):
        """
        Compute the cosine similarity matrix for the TF-IDF vectors.
        
        Returns:
            numpy.ndarray: A matrix of pairwise cosine similarity scores between movies.
        """
        # Calculate pairwise cosine similarity between all TF-IDF vectors
        return cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def recommend_movies(self, movie_title, num_recommendations=5):
        """
        Recommend movies similar to a given movie title based on content similarity.
        
        Args:
            movie_title (str): Title of the movie for which to generate recommendations.
            num_recommendations (int): Number of similar movies to recommend (default is 5).
        
        Returns:
            list: List of recommended movie titles.
        """
        # Create a mapping of movie titles to their indices
        movie_indices = pd.Series(self.movie_data.index, index=self.movie_data['title']).drop_duplicates()
        # Get the index of the movie corresponding to the input title
        idx = movie_indices.get(movie_title, None)

        # If the movie title is not found in the dataset, return an empty list
        if idx is None:
            return []

        # Retrieve similarity scores for the given movie
        similarity_scores = list(enumerate(self.similarity_matrix[idx]))
        # Sort movies by similarity score in descending order
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        # Select the top recommendations (excluding the input movie itself)
        similarity_scores = similarity_scores[1:num_recommendations + 1]

        # Extract indices of the recommended movies
        movie_indices = [i[0] for i in similarity_scores]
        # Return the titles of the recommended movies
        return self.movie_data['title'].iloc[movie_indices].tolist()
