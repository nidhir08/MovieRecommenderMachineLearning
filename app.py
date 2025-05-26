import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Step 1: Load the Netflix Shows dataset
shows = pd.read_csv("netflix-clustering/netflix_titles.csv")  

# Step 2: Preprocessing
shows['show_id'] = shows['show_id'].str.strip()  

# Step 3: Clean the data and create genres and cast features
def extract_genres(obj):
    try:
        L = obj.split(',')  
        return L
    except:
        return []

def extract_top_cast(obj):
    try:
        L = obj.split(',')[:3]  
        return L
    except:
        return []

# Apply the above functions
shows['genres'] = shows['listed_in'].apply(extract_genres)  # Genres in the 'listed_in' column
shows['cast'] = shows['cast'].apply(extract_top_cast)  # Cast from 'cast' column

# Step 4: Fill missing description with empty string
shows['description'] = shows['description'].fillna('')

# Step 5: Create 'tags' column (combining relevant text features for similarity calculation)
shows['tags'] = shows['description'] + ' ' + \
                 shows['genres'].apply(lambda x: ' '.join(x)) + ' ' + \
                 shows['cast'].apply(lambda x: ' '.join(x)) + ' ' + \
                 shows['director'].fillna('').apply(str)

# Step 6: Vectorization (TF-IDF)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(shows['tags'])

# Step 7: Similarity (Cosine similarity)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 8: Reverse mapping (for faster index lookup)
indices = pd.Series(shows.index, index=shows['title']).drop_duplicates()

# Step 9: Recommend function
def recommend_shows(title, cosine_sim=cosine_sim):
    if title not in indices:
        return pd.DataFrame()  # <<< return empty DataFrame, not list

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    show_indices = [i[0] for i in sim_scores]
    
    return shows.iloc[show_indices]


# Streamlit UI
st.title("📺 Movies & Show Recommender System")
st.write("Get Movies & Show recommendations based on genre, description, cast, and more!")

show_name = st.text_input("Enter a show/movie title (e.g., Stranger Things)")

if st.button("Recommend"):
    if show_name:
        results = recommend_shows(show_name)
        
        if results.empty:
            st.warning("Show not found in dataset.")
        else:
            st.subheader("Top 25 Similar Movies/Shows:")
            for _, show_row in results.iterrows():
                 show_title = show_row['title']
    
                 show_genres = ', '.join(show_row['genres']) if isinstance(show_row['genres'], list) and len(show_row['genres']) > 0 else 'N/A'
    
                 if isinstance(show_row['cast'], list) and len(show_row['cast']) > 0:
                  show_cast = ', '.join(show_row['cast'])
                 else:
                  show_cast = 'N/A'
    
                 show_rating = show_row['rating'] if 'rating' in show_row else 'N/A'
    
                 st.markdown(f"""
                **{show_title}**
                 - {show_row['description']}
                 - *Genres*: {show_genres}
                 - *Cast*: {show_cast}
                 - *Rating*: {show_rating}
                """)
