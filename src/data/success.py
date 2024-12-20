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
        # Load all the dataframes
        oscar = load_df_csv('data/processed_data/oscars_nominees.csv')
        goldenG = load_df_csv('data/processed_data/golden_globes_nominees.csv')
        filmfare = load_df_csv('data/processed_data/filmfare_nominees.csv')
        goldenP = load_df_csv('data/processed_data/golden_palms_nominees.csv')
        cesars = load_df_csv('data/processed_data/cesars_nominees.csv')
        asian_films = load_df_csv('data/processed_data/asian_films_nominees.csv')

        # Set the nomination column to 1
        oscar['nomination'] = 1
        goldenG['nomination'] = 1
        filmfare['nomination'] = 1
        goldenP['nomination'] = 1
        cesars['nomination'] = 1
        asian_films['nomination'] = 1
        awards = pd.concat([oscar,cesars,goldenG,filmfare,goldenP,asian_films],ignore_index=True)
        awards.drop_duplicates(subset=['Movie_name', 'Movie_release_date'], inplace=True)
        return awards

    except Exception as e:
        print(f"An error occurred while awards setup: {e}")

    
def merge_success_df(box_office_df, awards, ratings):
    """
    Merge all the dataframes together.
    """

    try:
        box_office_df['Movie_release_date'] = box_office_df['Movie_release_date'].apply(str)
        box_office_df['Wikipedia_movie_ID'] = box_office_df['Wikipedia_movie_ID'].astype(int)
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
            "Wikipedia_movie_ID",
            # "Actor_ethnicity",
            # "Movie_countries",
            "Movie_box_office_revenue",
            "Nomination"
        ]
        success_movies.columns = column_names
        # in column nomination, we replace 1 by True, otherwise False
        success_movies['Wikipedia_movie_ID'] = success_movies['Wikipedia_movie_ID'].astype(int)
        pd.set_option('future.no_silent_downcasting', True)
        success_movies['Nomination'] = success_movies['Nomination'].fillna(False)
        success_movies['Nomination'] = success_movies['Nomination'].replace(1.0,True)

        return success_movies
    except Exception as e:
        print(f"An error occurred while merging all dfs: {e}")

def drop_NaN_on_success(df):
    """
    Drop all the rows with NaN values.
    """
    try:
        df = df.dropna(subset=['Ratings'])
        return df
    except Exception as e:
        print(f"An error occurred while dropping NaN values: {e}")


def define_success(df, ratings_quantile=0.75, box_office_quantile=0.75):
    """
    Define success based on nominations, ratings and box office revenue.
    We take the movies with ratings and box office on the 3rd quartile and nominations.
    """
    try:
        # Define the threshold for ratings and box office revenue
        ratings_threshold = df['Ratings'].quantile(ratings_quantile)
        box_office_threshold = df['Movie_box_office_revenue'].quantile(box_office_quantile)

        # Define success based on the 3rd quartile for ratings and box office revenue
        df['Success'] = (df['Ratings'] > ratings_threshold) | (df['Nomination'] == 'True') | (df['Movie_box_office_revenue'] > box_office_threshold)
        df['Success'] = df['Success'].astype(int)
        # Print proportion of success movies
        number_successful_movies = df['Success'].sum()
        proportion_success_movies=number_successful_movies/len(df) * 100
        print("Proportion of success movies:", proportion_success_movies.round(2), "%")  
        return df
    except Exception as e:
        print(f"An error occurred while defining success: {e}")

def save_df_to_csv(df, csv_name):
    """
    Save the success DataFrame with a specified name, concate its name and path.
    """
    try:
        df.to_csv(f'data/processed_data/{csv_name}.csv',index=False, encoding='utf-8-sig')
        print("DataFrame saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving the DataFrame: {e}")

def merge_success_actors(success_df,actors_df):
    """
    Merge the success DataFrame with the actors DataFrame.
    """
    try:
        actors_df['Movie_release_date'] = actors_df['Movie_release_date'].apply(str)
        merge_on = ["Wikipedia_movie_ID", "Movie_name", "Movie_release_date"]
        success_actors_df = pd.merge(success_df, actors_df, on=merge_on, how='inner')
        # keep only the columns we need
        column_names = [
            "Movie_name",
            "Movie_release_date",
            "Ratings",
            "Wikipedia_movie_ID",
            "Movie_box_office_revenue",
            "Movie_countries",
            "Movie_languages",
            "Movie_runtime",
            "Nomination",
            "diversity",
            "Success",
            "actor_number",
        ]
        success_actors_df = success_actors_df[column_names]
        # columns = ['Movie_name', 'Movie_release_date', 'Ratings', 'Wikipedia_movie_ID', 'Movie_box_office_revenue', 'Nomination', 'Success',
        return success_actors_df
    except Exception as e:
        print(f"An error occurred while merging the success DataFrame with the actors DataFrame: {e}")