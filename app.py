from flask import Flask, jsonify, request
from recommender import ContentBasedRecommender

app = Flask(__name__)
recommender = ContentBasedRecommender('data/movies.csv')

@app.route('/recommend', methods=['GET'])
def recommend():
    movie_title = request.args.get('title', '')
    recommendations = recommender.recommend_movies(movie_title)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
