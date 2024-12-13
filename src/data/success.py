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

def ratings_setup(df_ratings, df_movie_names):
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

def box_office_setup(movie_metadata):
    """
    Get box office revenue from the given metadata.
    """
    try:
        boxO = movie_metadata[['Movie_name', 'Movie_release_date', 'Movie_box_office_revenue']]
        print("Columns taken successfully.")
        return boxO
    except Exception as e:
        print(f"An error occurred while creating boxO: {e}")

# def nominations_setup(oscar, goldenG,filmfare, goldenP,cesars):
#     """
#     Take all the nominations and add a column with ones.
#     Stack them together and drop duplicates.
#     """
#     try:
#         oscar['nomination'] = 1
#         goldenG['nomination'] = 1
#         filmfare['nomination'] = 1
#         goldenP['nomination'] = 1
#         cesars['nomination'] = 1
#         print("Columns added successfully.")

#         stacked_array = np.vstack([oscar,cesars,goldenG,filmfare,goldenP])
#         award = pd.DataFrame(stacked_array,columns=oscar.columns)
#         print("Columns stacked successfully.")

#         award['Movie_name'] = award['Movie_name'].apply(lambda x: ', '.join(sorted(x)) if isinstance(x, set) else x)
#         all_awards = award.drop_duplicates(subset=['Movie_name', 'Movie_release_date'])
#         print("Duplicates droped successfully.")
#         return all_awards

#     except Exception as e:
#         print(f"An error occurred while awards setup: {e}")

def nominations_setup(oscar, goldenG, filmfare, goldenP, cesars):
    """
    Take all the nominations and add a column with ones.
    Stack them together and drop duplicates.
    """
    try:
        for df in [oscar, goldenG, filmfare, goldenP, cesars]:
            if 'Movie_name' not in df.columns:
                raise KeyError("Missing column 'Movie_name' in one of the dataframes.")
            df['nomination'] = 1  # Add 'nomination' column
        print("Nomination columns added successfully.")

        stacked_array = pd.concat([oscar, cesars, goldenG, filmfare, goldenP], ignore_index=True)
        print("DataFrames stacked successfully.")

        stacked_array['Movie_name'] = stacked_array['Movie_name'].apply(
            lambda x: ', '.join(sorted(x)) if isinstance(x, set) else x
        )
        all_awards = stacked_array.drop_duplicates(subset=['Movie_name', 'Movie_release_date'])
        print("Duplicates dropped successfully.")
        return all_awards

    except KeyError as e:
        print(f"Error: Missing expected columns in nominations data: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while setting up awards: {e}")
        return pd.DataFrame()







