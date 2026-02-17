from flask import Flask, render_template, request
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

TMDB_API_KEY = "788186e43bd4aea1ce6c0d780e91740a"

movies = pd.read_csv("movies.csv")
movies = movies.head(5000)
movies = movies[['title', 'overview', 'genres']]
movies.fillna("", inplace=True)

movies["combined"] = movies["genres"] + " " + movies["overview"]

vectorizer = CountVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(movies["combined"])
similarity = cosine_similarity(matrix)

def get_poster(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    data = requests.get(url).json()

    if data["results"]:
        poster = data["results"][0]["poster_path"]
        if poster:
            return "https://image.tmdb.org/t/p/w500" + poster
    return ""

def recommend(movie):
    matches = movies[movies["title"].str.contains(movie, case=False, na=False)]

    if matches.empty:
        return []

    idx = matches.index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []

    for i in scores[1:6]:
        title = movies.iloc[i[0]].title
        poster = get_poster(title)
        results.append({"title": title, "poster": poster})

    return results

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []

    if request.method == "POST":
        movie = request.form["movie"]
        recommendations = recommend(movie)

    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
