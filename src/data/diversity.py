import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
#import matplotlib.pyplot as plt
#import seaborn as sns


# Set figure size for beautiful-jekyll compatibility
FIGURE_WIDTH = 800   # pixels
FIGURE_HEIGHT = 500  # pixels

def set_figsize(fig,width=FIGURE_WIDTH, height=FIGURE_HEIGHT):
    """Set the figure size for matplotlib plots."""
    fig.update_layout(width=width, height=height)

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

def ethnic_groups(actors_df,groups=16): #the groups determines which list we use. The difference is in the mixed, which is either broad or divided in 3 mixed groups
    ethnicities = actors_df['Actor_ethnicity'].unique().tolist() #list all the ethnicities we have
    if groups == 16:
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
        'Mixed Ethnicities/Global Diaspora': ['Eurasian','multiracial people', 'Afro-Asians', 'Métis', 'Asian Americans', 'Taiwanese Americans', 'Japanese Brazilians', 'White Africans of European ancestry', 'African Americans', 'British Pakistanis', 'Iranians in the United Kingdom', 'Black Britons', 'British Chinese', 'Anglo-Indian people', 'Brazilian Americans', 'History of the Jews in India','history of the Jews in India','Afro-Cuban','Vietnamese Americans','British Asian','Iranian Americans'],
        'Hispanic': ['Puerto Ricans', 'Spanish Americans', 'Cuban Americans', 'Mexican Americans', 'Stateside Puerto Ricans', 'Dominican Americans', 'Mexican Americans', 'Hispanic', 'Tejano', 'Spanish immigration to Mexico', 'Italian immigration to Mexico','Cuban Americans', 'Cubans', 'Mexican Americans', 'Bolivian American', 'Uruguayans', 'Mexicans', 'Argentines', 'Brazilians', 'Ecuadorian Americans', 'White Latin American', 'Hondurans', 'Venezuelans', 'Honduran Americans', 'Colombians', 'Chileans', 'Chileans in the United Kingdom', 'Salvadoran Americans','Venezuelan Americans', 'Latino', 'Panamanian Americans','Colombian Americans','Que viva el amor de Chaves',],
        'Caribbean': ['Haitian Americans', 'Bahamian Americans', 'Louisiana Creole people', 'British African-Caribbean people'],
         }
    elif groups == 18:
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
        'Mixed American': ['Asian Americans', 'Taiwanese Americans', 'African Americans', 'Brazilian Americans', 'Vietnamese Americans', 'Iranian Americans'],
        'Mixed European': ['Eurasian', 'White Africans of European ancestry','British Pakistanis', 'Iranians in the United Kingdom', 'Black Britons', 'British Chinese', 'Anglo-Indian people','British Asian'],
        'Mixed ': ['multiracial people', 'Afro-Asians', 'Métis', 'Japanese Brazilians',  'History of the Jews in India','history of the Jews in India','Afro-Cuban'],
        'Hispanic': ['Puerto Ricans', 'Spanish Americans', 'Cuban Americans', 'Mexican Americans', 'Stateside Puerto Ricans', 'Dominican Americans', 'Mexican Americans', 'Hispanic', 'Tejano', 'Spanish immigration to Mexico', 'Italian immigration to Mexico','Cuban Americans', 'Cubans', 'Mexican Americans', 'Bolivian American', 'Uruguayans', 'Mexicans', 'Argentines', 'Brazilians', 'Ecuadorian Americans', 'White Latin American', 'Hondurans', 'Venezuelans', 'Honduran Americans', 'Colombians', 'Chileans', 'Chileans in the United Kingdom', 'Salvadoran Americans','Venezuelan Americans', 'Latino', 'Panamanian Americans','Colombian Americans','Que viva el amor de Chaves',],
        'Caribbean': ['Haitian Americans', 'Bahamian Americans', 'Louisiana Creole people', 'British African-Caribbean people'],
         }
    
    ethnicity_to_group = {}
    for group, ethnicities in ethnic_groups.items():
        for ethnicity in ethnicities:
            ethnicity_to_group[ethnicity] = group

    classified_ethnicities = {ethnicity: ethnicity_to_group.get(ethnicity, 'Unknown') for ethnicity in ethnicities}
    actors_df['ethnic_group'] = actors_df['Actor_ethnicity'].map(ethnicity_to_group) #maps every ethnicity to its group
    return actors_df

def check_nan_Ethnicity(actors_df): #we want to check for nan ethnicity values
    actors_isnull = actors_df.isnull()
    nan_lines = actors_df[actors_df['ethnic_group'].isnull() == True]['Actor_ethnicity'] #identify what lines 
    nan_ethnicity = nan_lines.value_counts() # will count the nan values
    print(nan_ethnicity) #to make sure there are no nan values. If there are any, we can quickly identify what causes the problem
    return 

def naive_diversity(actors_df): #define a naive coefficient for diversity
    mov_div = actors_df.groupby('Wikipedia_movie_ID').agg(ethnicity_number=('ethnic_group','nunique'),actor_number=('Wikipedia_movie_ID','size')).reset_index() #groups the actors list by movie and counts the amount of actors per movie
    mov_div['naive_diversity']=mov_div['ethnicity_number']/mov_div['actor_number'] #our naive diversity score
    return mov_div

def ethnic_entropy(actors_df,mov_div,type = '*'): #the type is how we define diversity score
    ethn_count = actors_df.groupby(['Wikipedia_movie_ID','ethnic_group']).agg('size').reset_index(name='num_actors')                #groups by movie and ethnic group
    ethn_count = ethn_count.merge(mov_div[['Wikipedia_movie_ID','actor_number']], on='Wikipedia_movie_ID', how='left')              #merge onto movie df
    ethn_count['proportion'] = ethn_count['num_actors']/ethn_count['actor_number']                                                  #proportion of an ethnic group over total actors of that movie
    ethn_count['entropy'] = 1-ethn_count['proportion'] * np.log(ethn_count['proportion'])                                           #our formula for entropy, adding 1 so that we never have a null value 
    mov_div['max_entropy'] = mov_div['actor_number']*np.log(16)                                                                     #max entropy of a state with 16 possible ethnic_groups 
    entropy_by_movie = ethn_count.groupby('Wikipedia_movie_ID')['entropy'].sum().reset_index()                                      #sum the entropies for a single movie
    diversity_final = mov_div.merge(entropy_by_movie[['Wikipedia_movie_ID', 'entropy']], on='Wikipedia_movie_ID', how='left')       #merge everything together for final_df
    # diversity_final.to_csv('div_test.csv')
    if type == 'naive': 
        diversity_final['diversity'] = diversity_final['naive_diversity']
    elif type == 'entropy': 
        diversity_final['diversity'] = diversity_final['entropy']
    elif type == 'entropy_norm': #normalized the entropy
        diversity_final['diversity'] = diversity_final['entropy']/diversity_final['max_entropy']
    elif type == '*': #naive*entropy
        diversity_final['diversity']= diversity_final['naive_diversity']*diversity_final['entropy']
    elif type == '*norm': #naive*normalized entropy
        diversity_final['diversity']=diversity_final['naive_diversity']*diversity_final['entropy']/diversity_final['max_entropy'] 
    elif type == '+norm': #naive + normalized entropy
        diversity_final['diversity'] = diversity_final['diversity']=diversity_final['naive_diversity'] + diversity_final['entropy']/diversity_final['max_entropy'] 
    return diversity_final

def ethnic_entropy_old(actors_df,mov_div): #this is the old function, we don't use it
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

def drop_solo_actors(df): 
    """
    Drops all the rows with a single actor to avoid bias for diversity score.
    """
    df = df.dropna(subset=['diversity'])
    df = df[df["actor_number"] != 1]
    df = df.drop(columns='Actor_ethnicity').drop_duplicates(subset='Wikipedia_movie_ID')
    return df

def piechart(df,column,Title):
    ethnicity_counts = df[column].value_counts().reset_index()
    ethnicity_counts.columns = [column,'count']
    custom_colors = [
    "#A01812",  # Rouge foncé
    "#7A0A08",  # Rouge plus foncé
    "#3A0605",  # Rouge plus plus foncé
    "#6CA9B3",  # Bleu
    "#3F6475",  # Bleu foncé
    "#2A5563",  # Bleu plus foncé
    "#204F61",  # Bleu plus foncé 2
    "#284146",  # Bleu plus foncé 3
    "#1A3040",  # Bleu plus plus foncé
    # "#FFF8D3",  # Beige très clair
    "#FEF7D0",  # Beige clair
    "#FFF3BC",  # Beige clair jaune
    "#D8AC62",  # Beige
    "#C48530",  # Beige +
    "#B77526",  # Beige ++
    "#8A3D0C",  # Marron
    "#4C1508"   # Marron foncé
    ]
    fig = px.pie(ethnicity_counts, 
             names=column, 
             values='count', 
             title=Title,
             hover_name=column,
             color_discrete_sequence=custom_colors, # Allows the ethnicity to appear when hovering
             labels={column: column, 'Count': 'Number of Actors'})
    fig.update_traces(textinfo='none')
    fig.update_layout(paper_bgcolor="#FFF8D3")
    set_figsize(fig)
    fig.show()
    return fig

def plot_piechart():
    """Plot a piechart and save it as an html file."""
    actors_df = load_df('data/processed_data/clean_dataset.csv')
    title = "Actors ethnicities distribution"
    figure_ethnicities = piechart(actors_df,'Actor_ethnicity',title)
    pio.write_html(figure_ethnicities, file="./tests/ethnicities_piechart.html", auto_open=False)

# actors_diversity = ethnic_groups(actors_df)
# figure_ethnic_group = piechart(actors_diversity,'ethnic_group', 'Ethnic Groups') 
# pio.write_html(figure_ethnic_group, file="./tests/ethnic_groups_piechart.html", auto_open=False)