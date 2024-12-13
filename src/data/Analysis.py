import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
# from statsmodels.stats import diagnostic
from scipy import stats
# import networkx as nx
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# import plotly.express as px
import sys

### Import data
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

### Main visualisations

def plot_evolution(dataframe):
    diversity_by_year = dataframe.groupby(dataframe['Movie_release_date']).apply(lambda x: pd.Series({
        'average_diversity': x['diversity'].mean(),
        'std_diversity': x['diversity'].std()
    }))	

    plt.plot(diversity_by_year.index, diversity_by_year.average_diversity)
    plt.title('Average diversity score')
    plt.xlabel('Year')
    plt.ylabel('Diversity Score')
    plt.legend()
    plt.show()

def courbe_tendance(dataframe):
    filtered_df = dataframe[ethnicity_sucess_df['Movie_release_date'].between(1960, 2023)]
    diversity_by_year = filtered_df.groupby(filtered_df['Movie_release_date']).apply(
    lambda x: pd.Series({'average_diversity': x['diversity'].mean()})
)
    plt.plot(diversity_by_year.index, diversity_by_year['average_diversity'], label='Diversity Score')

    z = np.polyfit(diversity_by_year.index, diversity_by_year['average_diversity'], 1)  # Ajustement linéaire
    p = np.poly1d(z)
    plt.plot(diversity_by_year.index, p(diversity_by_year.index), "r--", label='Trend Line')

    plt.xlabel('Year')
    plt.ylabel('Diversity Score')
    plt.title('Average diversity score (1960-2023)')
    plt.legend()
    plt.show()

### Statistic Analysis

def mean_diversity (threshold_revenue, threshold_ratings, dataframe):
    dataframe['Success'] = (dataframe['Ratings'] >threshold_ratings ) | (dataframe['Nomination'] == 'True') | (dataframe['Movie_box_office_revenue'] > threshold_revenue)

    diversite_nomination_1=dataframe.loc[dataframe['Nomination'] == True]['diversity'].mean()
    diversite_box_1=dataframe.loc[dataframe['Movie_box_office_revenue']> threshold_revenue]['diversity'].mean()
    diversite_ratings_1=dataframe.loc[dataframe['Ratings'] > threshold_ratings ]['diversity'].mean()
    diversite_overall_1=dataframe.loc[dataframe['Success'] == True]['diversity'].mean()

    diversite_nomination_0=dataframe.loc[dataframe['Nomination'] == False]['diversity'].mean()
    diversite_box_0=dataframe.loc[dataframe['Movie_box_office_revenue']<= threshold_revenue]['diversity'].mean()
    diversite_ratings_0=dataframe.loc[dataframe['Ratings'] <= threshold_ratings ]['diversity'].mean()
    diversite_overall_0=dataframe.loc[dataframe['Success'] == False]['diversity'].mean()

    print(f"Average diversity for film nominated:{diversite_nomination_1:.4f}")
    print(f"Average diversity for film not nominated:{diversite_nomination_0:.4f}")

    print(f"Average diversity for film with high box revenue:{diversite_box_1:.4f}")
    print(f"Average diversity for film with lower box revenue:{diversite_box_0:.4f}")

    print(f"Average diversity for film with high ratings:{diversite_ratings_1:.4f}")
    print(f"Average diversity for film with lower ratings:{diversite_ratings_0:.4f}")

    print(f"Average diversity for film sucessful :{diversite_overall_1:.4f}")
    print(f"Average diversity for film less sucessful :{diversite_overall_0:.4f}") 
    
    # plot results.     
    categories = ['Nominated', 'Not Nominated']
    diversity_nomination = [diversite_nomination_1, diversite_nomination_0]
    categories_box = ['High Revenue', 'Low Revenue']
    diversity_box = [diversite_box_1, diversite_box_0]
    categories_ratings = ['High Ratings', 'Low Ratings']
    diversity_ratings = [diversite_ratings_1, diversite_ratings_0]
    categories_success = ['Successful', 'Not Successful']
    diversity_success = [diversite_overall_1, diversite_overall_0]
    # Create the figure and subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    # First plot: Nomination
    sns.barplot(x=categories,y=diversity_nomination,ax=axes[0])
    axes[0].set_title('Diversity: Nomination')
    axes[0].set_ylabel('Average Diversity')
    axes[0].set_ylim(0, 1)  # Adjust based on your diversity score range
    axes[0].set_xticks(range(len(categories)))
    axes[0].set_xticklabels(categories, rotation=30)

    # Second plot: Box Office Revenue
    sns.barplot(x=categories_box,y=diversity_box,ax=axes[1])
    axes[1].set_title('Diversity: Box Office Revenue')
    axes[1].set_xticks(range(len(categories_box)))
    axes[1].set_xticklabels(categories_box, rotation=30)

    # Third plot: Ratings
    sns.barplot(x=categories_ratings,y=diversity_ratings,ax=axes[2])
    axes[2].set_title('Diversity: Ratings')
    axes[2].set_xticks(range(len(categories_ratings)))
    axes[2].set_xticklabels(categories_ratings, rotation=30)

    # Fourth plot: Success
    sns.barplot(x=categories_success,y=diversity_success,ax=axes[3])
    axes[3].set_title('Diversity: Success')
    axes[3].set_xticks(range(len(categories_success)))
    axes[3].set_xticklabels(categories_success, rotation=30)

    # Adjust layout
    plt.tight_layout()
    plt.show()

def get_t_test(threshold_revenue, threshold_ratings,dataframe):
    dataframe['Success'] = (dataframe['Ratings'] >threshold_ratings ) | (dataframe['Nomination'] == 'True') | (dataframe['Movie_box_office_revenue'] > threshold_revenue)
    t_test_sucess=stats.ttest_ind(dataframe.loc[dataframe['Success'] == True]['diversity'], dataframe.loc[dataframe['Success'] == False]['diversity'])
    t_test_nomination=stats.ttest_ind(dataframe.loc[dataframe['Nomination'] == True]['diversity'], dataframe.loc[dataframe['Nomination'] == False]['diversity'])
    t_test_box=stats.ttest_ind(dataframe.loc[dataframe['Movie_box_office_revenue'] > threshold_revenue]['diversity'], dataframe.loc[dataframe['Movie_box_office_revenue'] <= threshold_revenue]['diversity'])
    t_test_ratings=stats.ttest_ind(dataframe.loc[dataframe['Ratings'] > threshold_ratings ]['diversity'], dataframe.loc[dataframe['Ratings'] <= threshold_ratings ]['diversity'])

    print(f"Nomination:{t_test_nomination}")
    print(f"Box revenue:{t_test_box}")
    print(f"Ratings:{t_test_ratings}")
    print(f"Success:{t_test_sucess}")

### Propensity score matching
def count_countries(countries_str):
    # Séparer les pays en fonction de la virgule et compter le nombre de pays
    countries = countries_str.split(',')  # Split par la virgule
    return len(countries)

def count_languages(languages_str):
    # Séparer les pays en fonction de la virgule et compter le nombre de pays
    languages = languages_str.split(',')  # Split par la virgule
    return len(languages)

############ final main
df = load_df('data/processed_data/clean_div_dataset.csv')
# df = df.dropna(subset=['Movie_box_office_revenue'])


############ Plots
plot_evolution(df)
courbe_tendance(df)
#---
mean_diversity (38119483, 7.5, df)
get_t_test(38119483, 7.5, df)
#---
mean_diversity (38119483, 6.9, df)
mean_diversity (38119483, 7.2, df)
mean_diversity (38119483, 8, df)

plt.scatter((6.9, 7.2, 7.5, 8),(0.5136,0.4806,0.4691,0.4600))
plt.title('evolution of diversity score modifying the thresold of ratinngs')
plt.show()
##
mean_diversity (23963802.0, 7.5, df)
mean_diversity (482083290, 7.5, df)