"""
---------------------------------------------------------
    File: load_awards_data.py
    Author: Francesco
    Description: < Load the wikipedia tables for several annual awards ceremony to 
    enhance our definition of succesfullness for movies>
"""
import pandas as pd
import os

"""
--------------------------------------------------------- """
## Import oscars best picture nominees
url = 'https://en.wikipedia.org/wiki/List_of_Academy_Award%E2%80%93nominated_films'
tables = pd.read_html(url)
oscars_awards_df = tables[0]
# We will focus on the film name and year of release to match it with our dataset
oscars_awards_df = oscars_awards_df[oscars_awards_df['Nominations']!=0][['Film','Year (Ceremony)']]

#now we normalise the Year column by only keeping the 4 first digits
oscars_awards_df['Year (Ceremony)'] = oscars_awards_df['Year (Ceremony)'].astype(str).str[:4]

# Give the same column names as in our cleaned dataset : 
oscars_awards_df.columns= ['Movie_name', 'Movie_release_date']

# Save the dataframe as a csv file in \processed_data directory
# save_path = r'..\..\data\processed_data\oscars_nominees.csv'
save_path = r'data\processed_data\oscars_nominees.csv'
oscars_awards_df.to_csv(save_path,index=False, encoding='utf-8-sig')


"""
--------------------------------------------------------- """
## Import golden globes best picture nominees
url = 'https://en.wikipedia.org/wiki/Golden_Globe_Award_for_Best_Motion_Picture_%E2%80%93_Drama'
tables = pd.read_html(url)
golden_globe_awards_df = pd.concat(tables[1:9],ignore_index=True)

# We will focus on the film name and year of release to match it with our dataset
golden_globe_awards_df = golden_globe_awards_df[['Year','Film']]

#now we normalise the Year column by only keeping the 4 first digits
golden_globe_awards_df.Year = golden_globe_awards_df.Year.astype(str).str[:4]

# Give the same column names as in our cleaned dataset : 
cols = list(golden_globe_awards_df.columns)
cols = [cols[1],cols[0]]
golden_globe_awards_df = golden_globe_awards_df[cols]
golden_globe_awards_df.columns= ['Movie_name', 'Movie_release_date']

# Save the dataframe as a csv file in \processed_data directory
# save_path = r'..\..\data\processed_data\golden_globes_nominees.csv'
save_path = r'data\processed_data\golden_globes_nominees.csv'
golden_globe_awards_df.to_csv(save_path,index=False, encoding='utf-8-sig')

"""
--------------------------------------------------------- """
## Import cesar best picture nominees
url = 'https://en.wikipedia.org/wiki/C%C3%A9sar_Award_for_Best_Film'
tables = pd.read_html(url)
cesar_awards_df = pd.concat(tables[2:7],ignore_index=True)
# We will focus on the film name and year of release to match it with our dataset
cesar_awards_df = cesar_awards_df[['Year','Original title']]

#now we normalise the Year column by only keeping the 4 first digits
cesar_awards_df.Year = cesar_awards_df.Year.astype(str).str[:4]

# Give the same column names as in our cleaned dataset : 
cols = list(cesar_awards_df.columns)
cols = [cols[1],cols[0]]
cesar_awards_df = cesar_awards_df[cols]
cesar_awards_df.columns= ['Movie_name', 'Movie_release_date']

# Save the dataframe as a csv file in \processed_data directory
# save_path = r'..\..\data\processed_data\cesars_nominees.csv'
save_path = r'data\processed_data\cesars_nominees.csv'
cesar_awards_df.to_csv(save_path,index=False, encoding='utf-8-sig')


"""
--------------------------------------------------------- """
## Import asian film award for best picture nominees
url = 'https://en.wikipedia.org/wiki/Asian_Film_Award_for_Best_Film'
tables = pd.read_html(url)
asian_film_awards_df = pd.concat(tables[2:4],ignore_index=True)

# We will focus on the film name and year of release to match it with our dataset
asian_film_awards_df = asian_film_awards_df[['Year','Original title']]

#now we normalise the Year column by only keeping the 4 first digits
asian_film_awards_df.Year = asian_film_awards_df.Year.astype(str).str[:4]

# Give the same column names as in our cleaned dataset : 
cols = list(asian_film_awards_df.columns)
cols = [cols[1],cols[0]]
asian_film_awards_df = asian_film_awards_df[cols]
asian_film_awards_df.columns= ['Movie_name', 'Movie_release_date']

# Save the dataframe as a csv file in \processed_data directory
# save_path = r'..\..\data\processed_data\asian_films_nominees.csv'
save_path = r'data\processed_data\asian_films_nominees.csv'
asian_film_awards_df.to_csv(save_path,index=False, encoding='utf-8-sig')

"""
--------------------------------------------------------- """
## Import filmfare award for best picture nominees
url = 'https://en.wikipedia.org/wiki/Filmfare_Award_for_Best_Film'
tables = pd.read_html(url)
filmfare_award_df = pd.concat(tables[1:9],ignore_index=True)

# We will focus on the film name and year of release to match it with our dataset
filmfare_award_df = filmfare_award_df[['Year','Film']]
filmfare_award_df = filmfare_award_df.dropna()

#now we normalise the Year column by only keeping the 4 first digits
filmfare_award_df.Year = filmfare_award_df.Year.astype(str).str[:4]

# Give the same column names as in our cleaned dataset : 
cols = list(filmfare_award_df.columns)
cols = [cols[1],cols[0]]
filmfare_award_df = filmfare_award_df[cols]
filmfare_award_df.columns= ['Movie_name', 'Movie_release_date']

# Save the dataframe as a csv file in \processed_data directory
# save_path = r"..\..\data\processed_data\filmfare_nominees.csv"
save_path = r"data\processed_data\filmfare_nominees.csv" # r as a raw string to avoir\f counted as an operation
filmfare_award_df.to_csv(save_path,index=False, encoding='utf-8-sig')

"""
--------------------------------------------------------- """
## Import golden palm award for best picture nominees
url = 'https://en.wikipedia.org/wiki/Palme_d%27Or#Winners'
tables = pd.read_html(url)
golden_palm_awards_df = tables[1]

# We will focus on the film name and year of release to match it with our dataset
# We will focus on the film name and year of release to match it with our dataset
golden_palm_awards_df = golden_palm_awards_df[['Year','Original title']]
indices_to_drop = [0,1,19,54,39,45,13,28,105]
golden_palm_awards_df = golden_palm_awards_df.drop(indices_to_drop)

#now we normalise the Year column by only keeping the 4 first digits
golden_palm_awards_df.Year = golden_palm_awards_df.Year.astype(str).str[:4]

# Give the same column names as in our cleaned dataset : 
cols = list(golden_palm_awards_df.columns)
cols = [cols[1],cols[0]]
golden_palm_awards_df = golden_palm_awards_df[cols]
golden_palm_awards_df.columns= ['Movie_name', 'Movie_release_date']
def clean_answer2(s):
    s = str(s).replace(' §', '').strip()
    return s

golden_palm_awards_df['Movie_name'] = golden_palm_awards_df['Movie_name'].apply(clean_answer2)

# Save the dataframe as a csv file in \processed_data directory
# save_path = r'..\..\data\processed_data\golden_palms_nominees.csv'
save_path = r'data\processed_data\golden_palms_nominees.csv'
golden_palm_awards_df.to_csv(save_path,index=False, encoding='utf-8-sig')
