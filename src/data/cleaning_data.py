# Some basic imports
import os
import pandas as pd
import numpy as np
import warnings
import ast
import re

def load_df(path_df):
    """
    Loads a DataFrame from a specified path.
    """
    try:
        df = pd.read_csv(path_df, sep='\t')
        print("DataFrame loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file at path '{path_df}' does not exist. Please check the file path.")
    except pd.errors.ParserError:
        print(f"Error: Failed to parse the file at '{path_df}'. Ensure it is in the correct tab-delimited format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def keep_info_of_interest(df_movie, df_character):
    """
    Rename the columns title of the specified DataFrame.
    Keep only columns of interest.
    Merge both df in one on matching features.

    return: df['Wikipedia_movie_ID', 'Movie_release_date', 'Actor_ethnicity', 'Movie_name', 'Movie_release_date', "Movie_runtime", "Movie_languages", 'Movie_countries']
    """
    try:
        new_column_names_m = [
            "Wikipedia_movie_ID",
            "Freebase_movie_ID",
            "Movie_name",
            "Movie_release_date",
            "Movie_box_office_revenue",
            "Movie_runtime",
            "Movie_languages",
            "Movie_countries",
            "Movie_genres"
        ]
        df_movie.columns = new_column_names_m

        new_column_names_c = [
        "Wikipedia_movie_ID",
        "Freebase_movie_ID",
        "Movie_release_date",
        "Character_name",
        "Actor_date_of_birth",
        "Actor_gender",
        "Actor_height_m",
        "Actor_ethnicity",
        "Actor_name",
        "Actor_age_at_movie_release",
        "Freebase_character_actor_map_ID",
        "Freebase_character_ID",
        "Freebase_actor_ID"
        ]
        df_character.columns = new_column_names_c
        print("Columns name changed successfully.")

        df_movie = df_movie['Wikipedia_movie_ID', 'Movie_name', 'Movie_release_date', "Movie_runtime", "Movie_languages", 'Movie_countries']
        df_character = df_character['Wikipedia_movie_ID', 'Movie_release_date', 'Actor_ethnicity']
        print("Columns selected successfully.")

        df_interest = pd.merge(df_character, df_movie, on=['Wikipedia_movie_ID', 'Movie_release_date'], how='inner')
        print("DataFrame merged successfully.")
        return df_interest
    except Exception as e:
        print(f"An error occurred while selecting and merging dfs: {e}")


def first_drop_nan(df_interest):
    """
    Drops all rows containing NaN values in the DataFrame.
    """
    try:
        df_cleaned = df_interest.dropna()
        print(f"Rows with NaN values dropped. Remaining rows: {len(df_cleaned)}")
        return df_cleaned
    except Exception as e:
        print(f"An error occurred while dropping NaN rows: {e}")

def rewrite_date(df_cleaned):
    """
    Write the date in one single way.
    """
    try:
        df_cleaned['Movie_release_date'] = df_cleaned['Movie_release_date'].astype(str).str[:4]
        print("Released Date column changed successfully.")
    except Exception as e:
        print(f"An error occurred while modifying the release date column: {e}")

    return df_cleaned

import pandas as pd

def replace_ethnicity_codes(df_cleaned, mapping_file_path):
    """
    Replaces ethnicity codes with corresponding ethnicity labels in a DataFrame.
    """
    def fb_to_label(freebase_id, conversion_table):
        """Maps a freebase ID to its label using the conversion table."""
        if freebase_id in conversion_table.index:
            return conversion_table.loc[freebase_id, 'label']
        else:
            return None
    
    try:
        fb_wiki_gen = pd.read_csv(mapping_file_path, sep='\t')
        fb_wiki_gen.set_index('freebase_id', inplace=True)

        # Filter the mapping table to only include relevant ethnicities
        ethnicities = df_cleaned['Actor_ethnicity'].unique()
        fb_wiki_ethnic = fb_wiki_gen.loc[fb_wiki_gen.index.isin(ethnicities)]

        df_cleaned = df_cleaned.copy()

        # Replace ethnicity codes with labels
        df_cleaned['Actor_ethnicity'] = df_cleaned['Actor_ethnicity'].apply(
            fb_to_label, conversion_table=fb_wiki_ethnic
        )

        # Drop rows where ethnicity is still NaN
        df_cleaned = df_cleaned.dropna(subset=['Actor_ethnicity'])
        df_cleaned['Actor_ethnicity'] = df_cleaned['Actor_ethnicity'].astype(str)

        print("Ethnicity column replaced successfully.")
        return df_cleaned

    except FileNotFoundError:
        print(f"Error: The mapping file '{mapping_file_path}' does not exist. Please check the file path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def replace_longest_ethnicity_with_eurasian(df_cleaned):
    """
    Finds the ethnicity value with the highest string length in the 'Actor_ethnicity' column
    and replaces it with 'Eurasian'.
    """
    try:
        # Ensure the column exists and convert to string for safety
        if 'Actor_ethnicity' not in df_cleaned.columns:
            raise KeyError("The 'Actor_ethnicity' column is not present in the DataFrame.")
        
        df_cleaned['Actor_ethnicity'] = df_cleaned['Actor_ethnicity'].astype(str)

        # Find the ethnicity with the longest string length
        longest_ethnicity = max(df_cleaned['Actor_ethnicity'], key=len)

        # Replace the longest ethnicity with 'Eurasian'
        df_cleaned['Actor_ethnicity'] = df_cleaned['Actor_ethnicity'].replace(longest_ethnicity, 'Eurasian')

        print("Replacement complete. Updated DataFrame:")
        return df_cleaned

    except Exception as e:
        print(f"An error occurred: {e}")
        return df_cleaned

def country_language_dico_clean(df_clean):
    """
    For country and language columns:
    - Removing rows where the dictionaries are empty.
    - Extracting only the values from the dictionaries and storing them as sets.
    """
    try:
        # Drop the rows where the dictionaries are empty
        df_clean = df_clean[df_clean['Movie_countries'].apply(lambda x: isinstance(x, dict) and len(x) > 0)]
        df_clean = df_clean[df_clean['Movie_languages'].apply(lambda x: isinstance(x, dict) and len(x) > 0)]
        print("Empty dictionaries removed successfully.")

        # Extract only the country and language names
        df_clean.loc[:, 'Movie_countries'] = df_clean['Movie_countries'].apply(
            lambda x: set(x.values()) if isinstance(x, dict) else set()
        )
        df_clean.loc[:, 'Movie_languages'] = df_clean['Movie_countries'].apply(
            lambda x: set(x.values()) if isinstance(x, dict) else set()
        )
        print("Values from dico extracted successfully.")

        return df_clean
    except Exception as e:
        print(f"An error occurred: {e}")

def remove_duplicates(df_clean):
    """
    Removes exact duplicate rows from the DataFrame.
    """
    try:
        df_cleaned = df_clean.drop_duplicates()
        print("Duplicates removed successfully.")
        return df_cleaned
    except Exception as e:
        print(f"An error occurred while removing duplicates: {e}")
        return df_clean

def main(
    movie_file_path, 
    character_file_path, 
    ethnicity_mapping_file_path
):
    """
    Main function to process movie and character data.
    """
    try:
        # Load movie and character data
        print("Loading movie and character data...")
        df_movie = load_df(movie_file_path)
        df_character = load_df(character_file_path)

        # Keep only information of interest
        print("Selecting and merging columns of interest...")
        df_interest = keep_info_of_interest(df_movie, df_character)

        # Drop rows with NaN values
        print("Dropping rows with NaN values...")
        df_cleaned = first_drop_nan(df_interest)

        print(df_interest.head())

        # Standardize the release date format
        print("Rewriting release dates...")
        df_cleaned = rewrite_date(df_cleaned)

        # Replace ethnicity codes with corresponding labels
        print("Replacing ethnicity codes with labels...")
        df_cleaned = replace_ethnicity_codes(df_cleaned, ethnicity_mapping_file_path)

        # Replace the longest ethnicity with 'Eurasian'
        print("Replacing the longest ethnicity with 'Eurasian'...")
        df_cleaned = replace_longest_ethnicity_with_eurasian(df_cleaned)

        # Clean country and language columns
        print("Cleaning country and language columns...")
        df_cleaned = country_language_dico_clean(df_cleaned)

        # Remove duplicate rows
        print("Removing duplicate rows...")
        df_cleaned = remove_duplicates(df_cleaned)

        print("Data processing completed successfully.")
        return df_cleaned

    except Exception as e:
        print(f"An error occurred during the data processing pipeline: {e}")
        return None