# Some basic imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from statsmodels.stats import diagnostic
from scipy import stats
import networkx as nx
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.express as px
import plotly.graph_objects as go

background_color = ["#FFF8D3"] # website background color

def color_palette(color):
    """take a color name as an argument and return the color in hex format"""
    "#A01812",  # Rouge foncé
    "#7A0A08",  # Rouge plus foncé
    "#3A0605",  # Rouge plus plus foncé
    "#6CA9B3",  # Bleu
    "#3F6475",  # Bleu foncé
    "#2A5563",  # Bleu plus foncé
    "#204F61",  # Bleu plus foncé 2
    "#284146",  # Bleu plus foncé 3
    "#1A3040",  # Bleu plus plus foncé
    "#FFF8D3",  # Beige très clair
    "#FEF7D0",  # Beige clair
    "#FFF3BC",  # Beige clair jaune
    "#D8AC62",  # Beige
    "#C48530",  # Beige +
    "#B77526",  # Beige ++
    "#8A3D0C",  # Marron
    "#4C1508"   # Marron foncé
    # Create a DataFrame with color names and hex codes
    colors_df = pd.DataFrame({
        'name': ['dark_red', 'darker_red', 'darkest_red',
                'blue', 'dark_blue', 'darker_blue', 'darker_blue2', 'darker_blue3', 'darkest_blue',
                'light_beige', 'light_beige2', 'light_yellow_beige',
                'beige', 'beige_plus', 'beige_plus_plus',
                'brown', 'dark_brown'],
        'hex': ['#A01812', '#7A0A08', '#3A0605',
                '#6CA9B3', '#3F6475', '#2A5563', '#204F61', '#284146', '#1A3040',
                '#FFF8D3', '#FEF7D0', '#FFF3BC',
                '#D8AC62', '#C48530', '#B77526',
                '#8A3D0C', '#4C1508']
    })
    # Return the hex code for the specified color
    color = colors_df[colors_df['name'] == color]['hex'].values[0]
    return [color]
    
def background_color(fig):
    """
    Set the background color of the plotly figure.
    """
    fig.update_layout(paper_bgcolor=background_color)

def plot_histogram(variable,parameter,color,title, html_output):
    """
    Plots a histogram of the specified variable as interactive
    plotly figure. Save it in html format. With the right background for the website.
    """
    fig = px.histogram(variable, x= parameter, title=title, color_discrete_sequence=color)
    fig.update_layout(paper_bgcolor="#FFF8D3") # website background color
    fig.write_html(html_output)
    fig.show()

def plot_evolution_basic(dataframe):
    """
    Plot the evolution of the average diversity score over the years.
    """
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

def plot_evolution(dataframe):
    diversity_by_year = dataframe.groupby(dataframe['Movie_release_date']).apply(lambda x: pd.Series({
        'average_diversity': x['diversity'].mean(),
        'std_diversity': x['diversity'].std()
    }))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=diversity_by_year.index,
        y=diversity_by_year.average_diversity,
        mode='lines+markers',
        name='Average Diversity',
        line=dict(color='#A01812'),
        marker=dict(size=8)
    ))

    # Add a title and axis labels
    fig.update_layout(
        title="Evolution of the diversity score overtime for the reduced dataset",
        xaxis_title="Year",
        yaxis_title="Diversity Score",
        legend_title="Metrics",
        template="plotly_white"
    )
    # Set the background color
    background_color(fig)
    # save it in html for the website
    fig.write_html("evolution_overtime.html")
    # Show the interactive plot
    fig.show()