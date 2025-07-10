An interactive Movie Recommender app built with Streamlit and Scikit-learn. Get movie suggestions by genre or find similar movies using TF-IDF and cosine similarity. Features a modern UI, fast performance, and genre-based filtering. Just run the app and start discovering your next favorite film!
🎬 Movie Recommender System
This repository contains a sleek, interactive Movie Recommendation System built using Python, Streamlit, and Scikit-learn. It recommends movies based on either:

🎭 Selected Genre: Pick a genre and instantly get a list of top matching movies.

🎞️ Movie Similarity: Enter a movie name to discover similar films using TF-IDF and cosine similarity on genres.

✨ Features
🔍 Dual Recommendation Modes (by Genre or Movie Title)

⚡ Fast, Responsive UI with CSS Animations

📊 Content-Based Filtering using TF-IDF Vectorization

🎨 Modern Gradient UI with interactive cards

📂 Data caching for optimized performance

📁 Requirements
Python 3.8+

streamlit, pandas, scikit-learn

🚀 How to Run
bash
Copy
Edit
pip install -r requirements.txt
streamlit run movie_recommender.py
Note: Ensure movie.csv is in the same directory as the script.
