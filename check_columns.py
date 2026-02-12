import pandas as pd

# Load the datasets
credits = pd.read_csv("C:\\Movie Recomendation\\Movie dataset\\tmdb_5000_credits.csv")
movies = pd.read_csv("C:\\Movie Recomendation\\Movie dataset\\tmdb_5000_movies.csv")

print("Columns in credits.csv:")
print(credits.columns.tolist())
print("\nColumns in movies.csv:")
print(movies.columns.tolist())
