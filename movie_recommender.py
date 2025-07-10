import streamlit as st
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("movie.csv")
    df['genres'] = df['genres'].fillna('').str.replace('|', ' ')
    df['genres_list'] = df['genres'].str.split()
    return df

df = load_data()

# --- TF-IDF Vectorization ---
@st.cache_resource
def get_cosine_sim_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genres'])
    return linear_kernel(tfidf_matrix, tfidf_matrix)

cosine_sim = get_cosine_sim_matrix(df)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# --- Recommendation Functions ---
def recommend_by_genre(genre):
    genre_movies = df[df['genres'].str.contains(genre, case=False)]
    titles = genre_movies['title'].dropna().tolist()
    random.shuffle(titles)
    return titles[:5]

def recommend_by_movie(title):
    if title not in indices:
        return ["Movie not found!"]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:20]  # top 20 similar
    random.shuffle(sim_scores)
    movie_indices = [i[0] for i in sim_scores[:5]]
    return df['title'].iloc[movie_indices].tolist()

# --- Autocomplete Function ---
def get_movie_suggestions(query, max_suggestions=10):
    if not query:
        return []
    
    # Filter movies that contain the query (case-insensitive)
    suggestions = df[df['title'].str.contains(query, case=False, na=False)]['title'].tolist()
    return suggestions[:max_suggestions]

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Movie Recommender", 
    page_icon="🎬", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# MODERN CSS WITH ANIMATIONS AND FULL-SCREEN OPTIMIZATION
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 0;
            overflow-x: hidden;
        }
        
        /* Remove Streamlit branding and margins */
        .stApp {
            background: transparent;
        }
        
        /* Main container */
        .main .block-container {
            padding: 1rem 1rem;
            max-width: 100%;
            margin: 0 auto;
        }
        
        /* Animated background */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            z-index: -1;
            animation: gradientShift 8s ease-in-out infinite;
        }
        
        @keyframes gradientShift {
            0%, 100% { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            50% { 
                background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
            }
        }
        
        /* Header styles */
        h1 {
            font-size: clamp(2.5rem, 5vw, 4rem) !important;
            font-weight: 700 !important;
            text-align: center !important;
            color: white !important;
            margin: 0 0 1rem 0 !important;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
            animation: fadeInDown 1s ease-out;
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Subtitle */
        .subtitle {
            text-align: center;
            color: rgba(255, 255, 255, 0.8);
            font-size: 1.1rem;
            margin-bottom: 3rem;
            animation: fadeIn 1.2s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        /* Radio buttons */
        .stRadio > div {
            display: flex;
            justify-content: center;
            gap: 2rem;
            flex-wrap: wrap;
            margin-bottom: 2rem;
        }
        
        .stRadio > div > label {
            background: rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 15px !important;
            padding: 1rem 2rem !important;
            border: 2px solid rgba(255, 255, 255, 0.2) !important;
            color: white !important;
            font-weight: 500 !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            min-width: 200px !important;
            text-align: center !important;
        }
        
        .stRadio > div > label:hover {
            background: rgba(255, 255, 255, 0.2) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2) !important;
        }
        
        .stRadio > div > label[data-testid="stRadio"] {
            background: rgba(255, 255, 255, 0.25) !important;
            border-color: rgba(255, 255, 255, 0.5) !important;
        }
        
        /* Enhanced Select boxes */
        .stSelectbox > div > div  {
            background: rgba(255, 255, 255, 0.1) !important;
            border-radius: 15px !important;
            color: white !important;
            backdrop-filter: blur(10px) !important;
        }
        
        /* Suggestions container */
        .suggestions-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            margin-top: -10px;
            margin-bottom: 1rem;
            max-height: 300px;
            overflow-y: auto;
            animation: slideInDown 0.3s ease-out;
            position: relative;
            z-index: 10;
        }
        
        .suggestion-item {
            padding: 0.8rem 1.2rem;
            color: white;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            cursor: pointer;
            transition: all 0.2s ease;
            font-weight: 500;
            display: block;
            text-decoration: none;
            background: none;
            border: none;
            width: 100%;
            text-align: left;
            font-size: 1rem;
            font-family: inherit;
        }
        
        .suggestion-item:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: translateX(5px);
        }
        
        .suggestion-item:last-child {
            border-bottom: none;
        }
        
        @keyframes slideInDown {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24) !important;
            color: white !important;
            border: none !important;
            border-radius: 50px !important;
            padding: 1rem 3rem !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            transition: background 0.3s ease, color 0.3s ease !important;
            box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3) !important;
            width: 100% !important;
            max-width: 300px !important;
            margin: 2rem auto !important;
            display: block !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #f0932b, #eb4d4b) !important;
            color: #fffbe6 !important;
        }

        
        /* Loading spinner */
        .stSpinner > div {
            border-color: rgba(255, 255, 255, 0.3) !important;
            border-top-color: white !important;
        }
        
        /* Success message */
        .stSuccess {
            background: rgba(46, 204, 113, 0.1) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 15px !important;
            border: 1px solid rgba(46, 204, 113, 0.3) !important;
            color: white !important;
            padding: 1rem !important;
            text-align: center !important;
            animation: bounceIn 0.6s ease-out !important;
        }
        
        @keyframes bounceIn {
            0% {
                opacity: 0;
                transform: scale(0.3);
            }
            50% {
                opacity: 1;
                transform: scale(1.05);
            }
            70% {
                transform: scale(0.9);
            }
            100% {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        /* Movie cards */
        .movie-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 1.5rem;
            margin: 0.5rem;
            text-align: center;
            color: white;
            transition: all 0.3s ease;
            animation: slideInUp 0.6s ease-out;
            animation-fill-mode: both;
            min-height: 120px; /* Ensures cards don’t look too small */
            height: auto;       /* Allows dynamic height based on content */
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .movie-card:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }
        
        .movie-card h4 {
            color: white;
            margin: 0;
            font-size: 1.1rem;
            font-weight: 600;
        }
        
        /* Input container for better spacing */
        .input-container {
            margin-bottom: 2rem;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem 0.5rem;
            }
            
            .stRadio > div {
                flex-direction: column;
                gap: 1rem;
            }
            
            .stRadio > div > label {
                min-width: auto !important;
            }
            
            .stSelectbox > div > div,
            .stTextInput > div > div > input {
                padding: 1rem !important;
                font-size: 1rem !important;
            }
        }

        
        /* Animation delays for staggered effect */
        .movie-card:nth-child(1) { animation-delay: 0.1s; }
        .movie-card:nth-child(2) { animation-delay: 0.2s; }
        .movie-card:nth-child(3) { animation-delay: 0.3s; }
        .movie-card:nth-child(4) { animation-delay: 0.4s; }
        .movie-card:nth-child(5) { animation-delay: 0.5s; }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.5);
        }

    </style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("<h1>🎬 Movie Recommender</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover your next favorite movie with AI-powered recommendations</div>', unsafe_allow_html=True)

# RECOMMENDATION MODE
mode = st.radio("Select Mode", ["🎭 Recommend by Genre", "🎞️ Recommend by Movie"], horizontal=True)

# ===== GENRE-BASED RECOMMENDATION =====
if mode == "🎭 Recommend by Genre":
    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    # Only keep clean, title-cased genres
    genres = sorted(set(g.title() for genres in df['genres_list'] for g in genres if g.isalpha()))
    selected_genre = st.selectbox("🎭 Choose a genre:", ["Please select a genre"] + genres)

    st.markdown('</div>', unsafe_allow_html=True)

    if selected_genre != "Please select a genre" and st.button("🔍 Discover Movies"):
        with st.spinner("Finding perfect matches..."):
            recommendations = recommend_by_genre(selected_genre)
            st.success(f"✨ Top movies in {selected_genre}")

            # Display movies in a responsive grid
            cols = st.columns([1, 1, 1, 1, 1])
            for i, movie in enumerate(recommendations):
                with cols[i]:
                    st.markdown(f'''
                        <div class="movie-card">
                            <h4>🎬 {movie}</h4>
                        </div>
                    ''', unsafe_allow_html=True)


# ===== MOVIE-BASED RECOMMENDATION =====

elif mode == "🎞️ Recommend by Movie":
    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    # Get all movie titles
    movie_titles = df['title'].dropna().unique().tolist()

    # Autocomplete input in the same field
    movie_input = st.selectbox(
        "🎞️ Type or select a movie:",
        options=[""] + movie_titles,
        index=0,
        placeholder="Start typing a movie name...",
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # Use the movie input for recommendations
    if movie_input and movie_input != "" and st.button("🎯 Find Similar Movies"):
        with st.spinner("Analyzing movie DNA..."):
            recommendations = recommend_by_movie(movie_input)

            if recommendations[0] != "Movie not found!":
                st.success(f"✨ Movies similar to {movie_input}")

                # Display movies in a responsive grid
                cols = st.columns([1, 1, 1, 1, 1])
                for i, movie in enumerate(recommendations):
                    with cols[i]:
                        st.markdown(f'''
                            <div class="movie-card">
                                <h4>🎬 {movie}</h4>
                            </div>
                        ''', unsafe_allow_html=True)
            else:
                st.error("🚫 Movie not found! Please select a movie from the list or try typing a different movie name.")
