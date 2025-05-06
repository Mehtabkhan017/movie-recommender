import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
from io import BytesIO
import time

# ------------------- Load and Preprocess Data -------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data.csv")
        # Basic data validation
        if not all(col in df.columns for col in ["title", "genres", "averageRating", "type", "releaseYear"]):
            st.error("Required columns missing in the dataset")
            return pd.DataFrame()
            
        df.dropna(subset=["title", "genres", "averageRating", "type", "releaseYear"], inplace=True)
        df["genres"] = df["genres"].fillna("Unknown").astype(str)
        
        # Create more comprehensive combined features
        df["combined_features"] = (
            df["title"].fillna("") + " " + 
            df["genres"].fillna("") + " " + 
            df["type"].fillna("") + " " +
            df["releaseYear"].astype(str) + " " +
            df["averageRating"].astype(str)
        )
        
        # Add placeholder for missing posters
        df["poster"] = df.get("poster", "https://via.placeholder.com/150x225?text=No+Poster")
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# ------------------- Recommendation Logic -------------------
def get_recommendations(movie_title, df, top_n=5):
    try:
        df = df.reset_index(drop=True)

        # Enhanced TF-IDF with more parameters
        tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Consider bi-grams
            min_df=2,            # Ignore terms that appear in less than 2 movies
            max_df=0.8           # Ignore terms that appear in more than 80% of movies
        )
        tfidf_matrix = tfidf.fit_transform(df['combined_features'])

        # Use linear_kernel for better performance with large datasets
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        if movie_title not in df["title"].values:
            return pd.DataFrame()

        idx = df[df['title'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

        movie_indices = [i[0] for i in sim_scores if i[0] < len(df)]
        return df.iloc[movie_indices]
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return pd.DataFrame()

# ------------------- Movie Filtering Logic -------------------
def filter_movies(df, selected_genres, selected_types, rating_range, year_range):
    try:
        filtered_df = df.copy()

        if selected_genres:
            filtered_df = filtered_df[filtered_df["genres"].apply(
                lambda g: any(genre.strip().lower() in g.lower().split(', ') for genre in selected_genres)
            )]

        if selected_types:
            filtered_df = filtered_df[filtered_df["type"].str.lower().isin([t.lower() for t in selected_types])]

        filtered_df = filtered_df[
            (filtered_df["averageRating"] >= rating_range[0]) &
            (filtered_df["averageRating"] <= rating_range[1]) &
            (filtered_df["releaseYear"] >= year_range[0]) &
            (filtered_df["releaseYear"] <= year_range[1])
        ]

        return filtered_df
    except Exception as e:
        st.error(f"Error filtering movies: {str(e)}")
        return df

# ------------------- Load Poster Image -------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_poster(url):
    try:
        if url.startswith('http'):
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content))
            return img
        return None
    except:
        return None

# ------------------- UI Layout & Logic -------------------
def main():
    # Improved page config with mobile-friendly settings
    st.set_page_config(
        page_title="🎥 Movie Recommender",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            'Get Help': 'https://github.com/yourusername/movie-recommender',
            'Report a bug': "https://github.com/yourusername/movie-recommender/issues",
            'About': "### 🎬 Advanced Movie Recommendation System\nPowered by Streamlit and scikit-learn"
        }
    )
    
    # Custom CSS for mobile responsiveness
    st.markdown("""
    <style>
        /* Main content responsive padding */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        /* Sidebar responsive adjustments */
        @media (max-width: 768px) {
            .sidebar .sidebar-content {
                width: 80% !important;
            }
            div[data-testid="stHorizontalBlock"] {
                flex-direction: column;
            }
        }
        
        /* Movie cards styling */
        .movie-card {
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #f8f9fa;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        /* Rating stars */
        .rating {
            color: #f5c518;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: #ff4c4c;'>🎬 Advanced Movie Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Discover your next favorite movie based on your preferences</p>", unsafe_allow_html=True)

    df = load_data()
    if df.empty:
        st.error("Failed to load data. Please check your data file.")
        return

    # Sidebar Filters with improved organization
    with st.sidebar:
        st.header("🔍 Filter Options")
        st.markdown("Customize your preferences below:")
        
        with st.expander("🎭 Genres & Types", expanded=True):
            genres = sorted(set(g.strip() for sublist in df["genres"].str.split(",") for g in sublist if g))
            selected_genres = st.multiselect(
                "Select genres",
                genres,
                key="genres_multiselect",
                help="Select one or more genres"
            )
            
            types = sorted(df["type"].dropna().unique())
            selected_types = st.multiselect(
                "Select movie types",
                types,
                default=["movie"],
                key="types_multiselect",
                help="Filter by movie type (movie, TV show, etc.)"
            )
        
        with st.expander("⭐ Rating & Year", expanded=True):
            rating_range = st.slider(
                "IMDb Rating range",
                0.0, 10.0, (5.0, 8.5), 0.1,
                key="rating_slider",
                help="Filter by IMDb rating range"
            )
            
            year_range = st.slider(
                "Release Year range",
                int(df["releaseYear"].min()), int(df["releaseYear"].max()), 
                (2000, 2020),
                key="year_slider",
                help="Filter by release year range"
            )
        
        with st.expander("⚙️ Recommendation Settings", expanded=False):
            top_n = st.slider(
                "Number of recommendations",
                3, 15, 5,
                key="top_n_slider",
                help="How many recommendations to show"
            )
        
        st.markdown("---")
        st.markdown("**About This App**")
        st.markdown("   Made by Mehtab khan")

    # Main content area
    st.markdown("---")
    
    # Filter movies with progress indicator
    with st.spinner("Finding movies that match your criteria..."):
        filtered_df = filter_movies(df, selected_genres, selected_types, rating_range, year_range)
    
    st.subheader(f"🎯 {len(filtered_df)} Movies Found Based on Filters")
    
    if len(filtered_df) == 0:
        st.warning("No movies found matching your criteria. Try adjusting your filters.")
        return
    
    # Select a movie with search functionality
    movie_title = st.selectbox(
        "🎬 Choose a movie for recommendations",
        filtered_df["title"].sort_values(),
        index=0,
        help="Select a movie to get similar recommendations"
    )
    
    # Display selected movie info in a card layout
    selected_row = filtered_df[filtered_df["title"] == movie_title].iloc[0]
    
    st.markdown("---")
    st.markdown(f"## {selected_row['title']} ({int(selected_row['releaseYear'])})")
    
    # Create columns for movie info and poster
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(f"**Genres:** {selected_row['genres']}")
        st.markdown(f"**Type:** {selected_row['type']}")
        st.markdown(f"**Rating:** <span class='rating'>{selected_row['averageRating']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Year:** {int(selected_row['releaseYear'])}")
        
        # Add a brief description if available in data
        if "description" in selected_row:
            st.markdown("**Description:**")
            st.write(selected_row["description"])
    
    with col2:
        poster = load_poster(selected_row.get("poster", ""))
        if poster:
            st.image(poster, caption=selected_row['title'], width=200)
        else:
            st.image("https://via.placeholder.com/200x300?text=No+Poster", width=200)
    
    st.markdown("---")
    st.subheader("🎯 Recommended Movies")
    
    # Get and display recommendations with progress indicator
    with st.spinner("Finding similar movies..."):
        recommended = get_recommendations(movie_title, filtered_df, top_n)
    
    if not recommended.empty:
        # Display recommendations in responsive grid
        cols = st.columns(2)  # 2 columns for mobile
        
        for idx, (_, row) in enumerate(recommended.iterrows()):
            with cols[idx % 2]:  # Alternate between columns
                with st.container():
                    st.markdown(f"### {row['title']} ({int(row['releaseYear'])})")
                    
                    # Try to load poster
                    poster = load_poster(row.get("poster", ""))
                    if poster:
                        st.image(poster, width=150)
                    
                    st.markdown(f"**Genres:** {row['genres']}")
                    st.markdown(f"**Type:** {row['type']}")
                    st.markdown(f"**Rating:** <span class='rating'>{row['averageRating']}</span>", unsafe_allow_html=True)
                    
                    # Add a "More Info" button that expands to show more details
                    with st.expander("More Info"):
                        if "description" in row:
                            st.write(row["description"])
                        else:
                            st.write("No additional information available.")
                    
                    st.markdown("---")
    else:
        st.warning("No similar recommendations found for this movie within the filters.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p>Built with ❤️ using Streamlit and scikit-learn</p>
        <p>Data from [Your Data Source] • [Privacy Policy] • [Terms of Use]</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------- Run the App -------------------
if __name__ == "__main__":
    main()