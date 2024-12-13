# Some basic imports
import pandas as pd
import numpy as np
import ast

def load_df(path_df):
    """
    Loads a DataFrame from a specified path.
    """
    try:
        df = pd.read_csv(path_df, sep='\t')
        # print("DataFrame loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file at path '{path_df}' does not exist. Please check the file path.")
    except pd.errors.ParserError:
        print(f"Error: Failed to parse the file at '{path_df}'. Ensure it is in the correct tab-delimited format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def keep_info_of_interest(df_ratings, df_movie_names):
    """
    Keep only columns of interest.
    Merge both df in one on matching feature.
    Drop one column and rename the rest.

    Output: df[['primaryTitle','startYear','averageRating']]
    """
    try:
        df_ratings_selected = df_ratings[['tconst','averageRating']]
        df_movie_names_selected = df_movie_names[['tconst','primaryTitle','startYear']]
        # print("Columns selected successfully.")

        df_interest = pd.merge(df_movie_names_selected,df_ratings_selected,on='tconst')
        # print("DataFrame merged successfully.")

        df_interest = df_interest.drop('tconst', axis=1)
        # print("Column drop successfully.")

        column_names = [
            "Movie_name",
            "Movie_release_date",
            "Ratings"
        ]
        df_interest.columns = column_names
        # print("Columns rename successfully.")
        return df_interest
    except Exception as e:
        print(f"An error occurred while selecting and merging dfs: {e}")

# def to add one the nominees



