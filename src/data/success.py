# Some basic imports
import pandas as pd
import numpy as np
import ast

def load_df_tsv(path_df):
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


def load_df_csv(path_df):
    """
    Loads a DataFrame from a specified path.
    """
    try:
        df = pd.read_csv(path_df)
        # print("DataFrame loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file at path '{path_df}' does not exist. Please check the file path.")
    except pd.errors.ParserError:
        print(f"Error: Failed to parse the file at '{path_df}'. Ensure it is in the correct tab-delimited format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def ratings_setup():
    """
    Keep only columns of interest.
    Merge both df in one on matching feature.
    Drop one column and rename the rest.

    Output: df[['primaryTitle','startYear','averageRating']]
    """
    try:
        ratings_path = 'data/raw_data/title.ratings.tsv'
        movie_names_path = 'data/raw_data/title.basics.tsv'
        df_ratings = load_df_tsv(ratings_path)
        df_movie_names = pd.read_csv(movie_names_path, sep='\t', low_memory=False)
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

def nominations_setup():
    """
    Take all the nominations and add a column with ones.
    Stack them together and drop duplicates.
    """
    try:
        oscar = load_df_csv('data/processed_data/oscars_nominees.csv')
        goldenG = load_df_csv('data/processed_data/golden_globes_nominees.csv')
        filmfare = load_df_csv('data/processed_data/filmfare_nominees.csv')
        goldenP = load_df_csv('data/processed_data/golden_palms_nominees.csv')
        cesars = load_df_csv('data/processed_data/cesars_nominees.csv')
        oscar['nomination'] = 1
        goldenG['nomination'] = 1
        filmfare['nomination'] = 1
        goldenP['nomination'] = 1
        cesars['nomination'] = 1
        awards = pd.concat([oscar,cesars,goldenG,filmfare,goldenP],ignore_index=True)
        awards.drop_duplicates(subset=['Movie_name', 'Movie_release_date'], inplace=True)
        return awards

    except Exception as e:
        print(f"An error occurred while awards setup: {e}")

def nominations_setup2(oscar, goldenG,filmfare, goldenP,cesars):
    """
    Take all the nominations and add a column with ones.
    Stack them together and drop duplicates.
    """
    try:
        oscar['nomination'] = 1
        goldenG['nomination'] = 1
        filmfare['nomination'] = 1
        goldenP['nomination'] = 1
        cesars['nomination'] = 1
        awards = pd.concat([oscar,cesars,goldenG,filmfare,goldenP],ignore_index=True)
        awards.drop_duplicates(subset=['Movie_name', 'Movie_release_date'], inplace=True)
        return awards

    except Exception as e:
        print(f"An error occurred while awards setup: {e}")

    
def merge_all_df(box_office_df, awards, ratings):
    """
    Merge all the dataframes together.
    """

    # try:
    box_office_df['Movie_release_date'] = box_office_df['Movie_release_date'].apply(str)
    df = pd.merge(ratings, box_office_df, on=['Movie_name', 'Movie_release_date'], how='inner')
    df = pd.merge(df, awards, on=['Movie_name'], how='outer')
    # in column nomination, we replace 1 by True, otherwise False
    pd.set_option('future.no_silent_downcasting', True)
    df['nomination']=df['nomination'].fillna(False)
    df['nomination']=df['nomination'].replace(1.0,True)
    success_movies=df.dropna(subset=['Ratings'])
    # drop unnecessary columns
    success_movies=success_movies.drop(columns=['Movie_release_date_y'])
    # rename columns
    column_names = [
        "Movie_name",
        "Movie_release_date",
        "Ratings",
        # "Wikipedia_movie_ID",
        # "Actor_ethnicity",
        # "Movie_countries",
        "Movie_box_office_revenue",
        "Nomination"
    ]
    success_movies.columns = column_names
    return success_movies
    # except Exception as e:
    #     print(f"An error occurred while merging all dfs: {e}")





