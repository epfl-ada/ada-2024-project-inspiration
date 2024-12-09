import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

def load_df(path_df):
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

def ethnic_groups(actors_df):
    ethnicities = actors_df['Actor_ethnicity'].unique().tolist()
    ethnic_groups = {
      'East Asian': ['Taiwanese', 'Koreans', 'Chinese Americans', 'Hongkongers', 'Chinese Singaporeans', 'Korean American', 'Chinese Filipino', 'Malaysian Chinese', 'Chinese Indonesians', 'Chinese Canadians', 'Chinese Jamaicans', 'Han Chinese people', 'Zhuang people', 'Manchu', 'Buryats','Gin people', 'Japanese people', 'Koryo-saram', 'Ryukyuan people'],
      'Southeast Asian': ['Filipino Americans', 'Vietnamese people', 'Lao people', 'Hmong American', 'Filipino people', 'Sri Lankan Tamils', 'Indonesian Americans', 'Filipino Australians', 'Kapampangan people', 'Ilocano', 'Filipino people of Spanish ancestry', 'Filipino mestizo', 'Aceh', 'Javanese', 'Thai people', 'Thai Americans', 'Thai Chinese people', 'Tibetan people','Sherpa'],
      'South Asian': ['Indian Americans', 'Indian', 'Ezhava', 'Malayali', 'Marathi people', 'Gujarati people', 'Bihari people', 'Nair', 'Bengali', 'Bunt', 'Sindhis', 'Tamil', 'Punjabis', 'British Indian', 'Telugu people', 'Kayastha', 'Brahmin caste', 'Kashmiri Pandit', 'Jaat', 'Tulu people', 'Marwari', 'Bhutia', 'Agrawal', 'Mudaliar', 'Sinhalese', 'Bengali Brahmins', 'Kannada people', 'Konkani people', 'Chettiar', 'Dalit', 'Dogra', 'Hazaras', 'Nepali Indian', 'Sikh', 'Niyogi', 'Kashmiri people', 'Rohilla', 'Indian diaspora in France', 'Muhajir', 'Hindu', 'Rajput','rajput', 'Khatri', 'Indian diaspora','Pakistanis'],
      'North African': ['Berber', 'Egyptians', 'Moroccans', 'Sudanese Arabs', 'Arab Mexican','Copts', 'Kabyle people', 'Arabs','Arabs in Bulgaria', 'History of the Jews in Morocco', 'Sephardi Jews'],
      'Sub-Saharan African': ['Yoruba people', 'Akan people', 'Xhosa people', 'Kikuyu', 'Mandinka people', 'Wolof people', 'Malagasy people', 'Sierra Leone Creole people', 'African people', 'Nigerian Americans', 'Somalis',  'White South Africans', 'Afrikaners'],
      'Middle Eastern': ['Jewish people', 'Armenians', 'American Jews', 'Iranian peoples', 'Iraqi Americans', 'Lebanese Americans', 'Syrian Americans', 'Israeli Americans', 'Lebanese', 'British Jews', 'culture of Palestine', 'Israeli Jews','Israelis', 'Lithuanian Jews', 'Assyrian people', 'Iranian Canadians', 'Persians', 'Arab Americans', 'Palestinian Americans', 'Armenians of Russia', 'Kurds', 'demographics of Iraq', 'Muslim', 'Pashtuns', 'Ashkenazi Jews', 'Lebanese immigration to Mexico', 'Turkish Americans', 'Tatars', 'Azerbaijanis'],
      'European descent American': ['Irish Americans', 'German Americans', 'Lithuanian American', 'Italian Americans', 'Danish Americans', 'Scottish Americans', 'English Americans', 'Spanish Americans', 'Swedish Americans', 'Armenian American','Finnish Americans', 'White Americans', 'Dutch Americans', 'Hungarian Americans', 'Italian Canadians', 'Americans', 'Serbian Americans', 'Polish Americans', 'French Canadians', 'Czech Americans', 'British Americans', 'Sicilian Americans', 'French Chilean', 'Albanian American', 'Austrian Americans', 'French Americans', 'Slovak Americans', 'Latvian American', 'Norwegian Americans', 'Portuguese Americans', 'Irish Canadians', 'Italian Brazilians', 'English Canadians', 'Cajun', 'Chilean American', 'Welsh Americans', 'Swedish Canadians', 'German Brazilians','German Canadians' ,'Greek Americans', 'Luxembourgish Americans','Danish Canadians', 'Russian Americans', 'Polish Canadians','Serbian Canadians','Romanian Americans',  'Bulgarian Canadians','Ukrainian Americans', 'Ukrainian Canadians', 'Dutch Canadians', 'Croatian Americans', 'Croatian Canadians','Slovene Americans', 'Acadians','Greek Canadians'],
      'Eastern European': ['Russians', 'Slovaks', 'Hungarians', 'Latvians', 'Belarusians', 'Soviet people', 'Czechs', 'Serbs of Croatia', 'Bosnians', 'Romani people', 'Greeks in South Africa', 'Romanichal', 'Aromanians', 'Swedish-speaking population of Finland', 'Baltic Russians', 'Transylvanian Saxons', 'Yugoslavs', 'Bulgarians', 'Bohemian', 'Hutsuls', 'Romanians', 'Serbs of Bosnia and Herzegovina','Slovenes', 'Slavs', 'Bosniaks', 'Poles', 'Serbs in North Macedonia', 'Croats', 'Albanians', 'Ossetians', 'Peoples of the Caucasus', 'Georgians','Ukrainians','Serbs in the United Kingdom','peoples of the Caucasus'],
      'Northern European': ['English people', 'Irish people', 'Welsh people', 'Swedes',  'Norwegians', 'Danes', 'Sámi peoples', 'Anglo-Irish people', 'Swedish-speaking population of Finland', 'Finns', 'Icelanders', 'Manx people','Parsi', 'White British', 'Irish migration to Great Britain', 'Poles in the United Kingdom'],
      'Western European': ['French', 'Germans', 'Spaniards', 'Austrians', 'Swiss', 'Galicians', 'Castilians', 'Portuguese', 'Belgians', 'Corsicans','Dutch'],
      'Southern European': ['Italians', 'Spaniards', 'Greeks', 'names of the Greeks', 'Dalmatian Italians', 'Catalan people', 'Greeks in South Africa',  'Greeks in the United Kingdom', 'Romani people in Spain', 'Greek Cypriots', 'Italians in the United Kingdom', 'Gibraltarian people', 'Basque people'],
      'Indigenous peoples of the Americas': ['Omaha people', 'Cherokee', 'Native Hawaiians', 'First Nations', 'Mohawk people', 'Inuit', 'Sioux', 'Lumbee', 'Cree', 'Apache', 'Haudenosaunee', 'Inupiat people', 'Cheyennes', 'Dene', 'Aymara', 'Oneida', 'Blackfoot Confederacy', 'Ojibwe', 'Indo-Canadians', 'Choctaw', 'Quebeckers', 'Ho-Chunk', 'Nez Perce'],
      'Oceania and Pacific Islander': ['Pacific Islander Americans', 'Samoan New Zealanders', 'Māori', 'Kiwi', 'Greek Australians', 'Australian Americans', 'Italian Australian','Australian Americans','Anglo-Celtic Australians', 'Swedish Australian', 'English Australian', 'Australians', 'Scottish Australian', 'Dutch Australian', 'Irish Australian', 'Croatian Australians', 'Polish Australians', 'Serbian Australians','Indian Australian'],
      'Mixed Ethnicities/Global Diaspora': ['Eurasian','freebase_id\n/m/017sq0    Eurasian\n/m/017sq0     Eurasia\nName: label, dtype: object','multiracial people', 'Afro-Asians', 'Métis', 'Asian Americans', 'Taiwanese Americans', 'Japanese Brazilians', 'White Africans of European ancestry', 'African Americans', 'British Pakistanis', 'Iranians in the United Kingdom', 'Black Britons', 'British Chinese', 'Anglo-Indian people', 'Brazilian Americans', 'History of the Jews in India','history of the Jews in India','Afro-Cuban','Vietnamese Americans','British Asian','Iranian Americans'],
      'Hispanic': ['Puerto Ricans', 'Spanish Americans', 'Cuban Americans', 'Mexican Americans', 'Stateside Puerto Ricans', 'Dominican Americans', 'Mexican Americans', 'Hispanic', 'Tejano', 'Spanish immigration to Mexico', 'Italian immigration to Mexico','Cuban Americans', 'Cubans', 'Mexican Americans', 'Bolivian American', 'Uruguayans', 'Mexicans', 'Argentines', 'Brazilians', 'Ecuadorian Americans', 'White Latin American', 'Hondurans', 'Venezuelans', 'Honduran Americans', 'Colombians', 'Chileans', 'Chileans in the United Kingdom', 'Salvadoran Americans','Venezuelan Americans', 'Latino', 'Panamanian Americans','Colombian Americans','Que viva el amor de Chaves',],
      'Caribbean': ['Haitian Americans', 'Bahamian Americans', 'Louisiana Creole people', 'British African-Caribbean people'],
    }
    ethnicity_to_group = {}
    for group, ethnicities in ethnic_groups.items():
        for ethnicity in ethnicities:
            ethnicity_to_group[ethnicity] = group

    classified_ethnicities = {ethnicity: ethnicity_to_group.get(ethnicity, 'Unknown') for ethnicity in ethnicities}
    actors_df['ethnic_group'] = actors_df['Actor_ethnicity'].map(ethnicity_to_group)
    return actors_df

def check_nan_Ethnicity(actors_df):
    actors_isnull = actors_df.isnull()
    nan_lines = actors_df[actors_df['ethnic_group'].isnull() == True]['Actor_ethnicity']
    nan_ethnicity = nan_lines.value_counts() # We don't have any NaN anymore
    print(nan_ethnicity)
    return 

def naive_diversity(actors_df):
    mov_div = actors_df.groupby('Wikipedia_movie_ID').agg(ethnicity_number=('ethnic_group','nunique'),actor_number=('Wikipedia_movie_ID','size')).reset_index()
    mov_div['naive_diversity']=mov_div['ethnicity_number']/mov_div['actor_number']
    return mov_div

def ethnic_entropy(actors_df,mov_div):
    ethn_count = actors_df.groupby(['Wikipedia_movie_ID','ethnic_group']).agg('size').reset_index(name='num_actors')
    tot_actors = ethn_count.groupby('Wikipedia_movie_ID')['num_actors'].transform('sum')

    ethn_count['proportion'] = ethn_count['num_actors']/tot_actors
    ethn_count['entropy'] = 1-ethn_count['proportion'] * np.log(ethn_count['proportion'])

    entropy_by_movie = ethn_count.groupby('Wikipedia_movie_ID')['entropy'].sum().reset_index()

    # Normalize entropy
    max_entropy = np.log(len(actors_df['ethnic_group'].unique()))  # Maximum possible entropy
    entropy_by_movie['normalized_entropy'] = entropy_by_movie['entropy'] / max_entropy

    # merge everything back together
    diversity_final = mov_div.merge(entropy_by_movie[['Wikipedia_movie_ID', 'normalized_entropy']], on='Wikipedia_movie_ID', how='left')
    diversity_final['diversity']= diversity_final['naive_diversity']*diversity_final['normalized_entropy']
    return diversity_final

def merge_on_movies(movies_df,actors_df):
    movies_final = pd.merge(movies_df,actors_df[['Wikipedia_movie_ID','diversity','actor_number']],on='Wikipedia_movie_ID', how='left')
    return movies_final

actors_df = load_df('data/processed_data/clean_dataset.csv')
actors_diversity = ethnic_groups(actors_df)
check_nan_Ethnicity(actors_diversity)
diversity = naive_diversity(actors_diversity)
diversity = ethnic_entropy(actors_df,diversity)
actors_df = merge_on_movies(actors_df,diversity)
print(actors_df)
