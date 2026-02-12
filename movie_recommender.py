import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Download required NLTK data
nltk.download('punkt')

# Initialize the stemmer
ps = PorterStemmer()

def load_data():
    # Load the datasets
    credits = pd.read_csv("C:\\Movie Recomendation\\Movie dataset\\tmdb_5000_credits.csv")
    movies = pd.read_csv("C:\\Movie Recomendation\\Movie dataset\\tmdb_5000_movies.csv")
    
    # Merge datasets
    credits_renamed = credits.rename(columns={'movie_id': 'id'})
    movies = movies.merge(credits_renamed, on='id')
    
    return movies

def preprocess_data(movies):
    # Select relevant columns from the merged dataset
    # Note: After merge, we have 'title_x' (from movies) and 'title_y' (from credits)
    # We'll use 'title_x' as our main title and 'title_y' as an alternative
    movies = movies[['id', 'title_x', 'title_y', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    # Rename columns for consistency
    movies = movies.rename(columns={
        'id': 'movie_id',
        'title_x': 'title',
        'title_y': 'original_title'
    })
    
    # Handle missing values
    movies.dropna(inplace=True)
    
    # Convert string representations of lists to actual lists
    import ast
    
    def convert_to_list(text):
        try:
            return [i['name'] for i in ast.literal_eval(text)]
        except:
            return []
    
    movies['genres'] = movies['genres'].apply(convert_to_list)
    movies['keywords'] = movies['keywords'].apply(convert_to_list)
    movies['cast'] = movies['cast'].apply(convert_to_list)
    
    # Extract director from crew
    def get_director(crew):
        try:
            for person in ast.literal_eval(crew):
                if person['job'] == 'Director':
                    return [person['name']]
            return []
        except:
            return []
    
    movies['crew'] = movies['crew'].apply(get_director)
    
    # Convert all text to lowercase
    movies['genres'] = movies['genres'].apply(lambda x: [i.lower() for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.lower() for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.lower() for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.lower() for i in x])
    
    # Create tags by combining relevant columns
    movies['tags'] = movies['overview'] + ' ' + \
                    movies['genres'].apply(' '.join) + ' ' + \
                    movies['keywords'].apply(' '.join) + ' ' + \
                    movies['cast'].apply(' '.join) + ' ' + \
                    movies['crew'].apply(' '.join)
    
    # Convert to lowercase and remove special characters
    movies['tags'] = movies['tags'].str.lower()
    
    return movies

def create_similarity_matrix(movies):
    # Initialize CountVectorizer
    cv = CountVectorizer(max_features=5000, stop_words='english')
    
    # Create the count matrix
    vectors = cv.fit_transform(movies['tags']).toarray()
    
    # Calculate cosine similarity
    similarity = cosine_similarity(vectors)
    
    return similarity

def recommend(movie, movies, similarity):
    # Find the index of the movie
    movie_index = movies[movies['title'] == movie].index[0]
    
    # Get similarity scores
    distances = similarity[movie_index]
    
    # Get top 5 most similar movies
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    # Print recommended movies
    print(f"\nMovies similar to '{movie}':\n" + "-"*50)
    for i, (index, score) in enumerate(movies_list, 1):
        print(f"{i}. {movies.iloc[index]['title']} (Similarity: {score:.2f})")

def main():
    print("Loading and preprocessing data...")
    movies = load_data()
    movies_processed = preprocess_data(movies)
    
    print("Creating similarity matrix...")
    similarity = create_similarity_matrix(movies_processed)
    
    # Example recommendation
    movie_name = "Avatar"  # You can change this to any movie in the dataset
    recommend(movie_name, movies_processed, similarity)
    
    # Interactive mode
    while True:
        print("\n" + "="*50)
        print("Movie Recommendation System")
        print("="*50)
        print("\nEnter a movie name (or 'q' to quit):")
        user_input = input("> ")
        
        if user_input.lower() == 'q':
            print("Goodbye!")
            break
            
        if user_input not in movies_processed['title'].values:
            print("Movie not found in the database. Please try another one.")
            continue
            
        recommend(user_input, movies_processed, similarity)

if __name__ == "__main__":
    main()
