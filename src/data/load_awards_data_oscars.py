"""
---------------------------------------------------------
    File: load_awards_data_oscars.py
    Author: Francesco
    Description: <load the wikipedia tables for the oscar nominees and store it in processed data>
"""
import pandas as pd

url = 'https://en.wikipedia.org/wiki/List_of_Academy_Award%E2%80%93nominated_films'
tables = pd.read_html(url)
# The interesting table is the 0: 
Academy_Award_df = tables[0]

# Academy_Award_df.shape | printed to see our table size

# We now want to obtain a df with Film name, Year in a clean manner
# A: remove films with 0 nominations
Academy_Awards_nominees = Academy_Award_df[Academy_Award_df['Nominations']!=0][['Film','Year (Ceremony)']]
Academy_Awards_nominees.shape
Academy_Awards_nominees.sample(5)

#now we normalise the Year column by only keeping the 4 first digits
Academy_Awards_nominees['Year (Ceremony)'] = Academy_Awards_nominees['Year (Ceremony)'].astype(str).str[:4]
Academy_Awards_nominees.sample(5)

Academy_Awards_nominees.columns = ['Movie_name', 'Movie_release_date']
Academy_Awards_nominees.sample(5)