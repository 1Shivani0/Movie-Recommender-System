import pandas as pd

# Load and merge datasets
credits = pd.read_csv("C:\\Movie Recomendation\\Movie dataset\\tmdb_5000_credits.csv")
movies = pd.read_csv("C:\\Movie Recomendation\\Movie dataset\\tmdb_5000_movies.csv")

# Merge datasets
credits_renamed = credits.rename(columns={'movie_id': 'id'})
merged = movies.merge(credits_renamed, on='id')

print("Columns in merged dataset:")
print(merged.columns.tolist())

print("\nFirst few rows to check for any suffix additions:")
print(merged[['id', 'title_x', 'title_y']].head())
