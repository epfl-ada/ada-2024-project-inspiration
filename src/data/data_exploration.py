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
    
def set_background_color(fig):
    """
    Set the background color of the plotly figure.
    """
    fig.update_layout(paper_bgcolor=background_color[0])

def plot_histogram(variable,parameter,color,title, html_output):
    """
    Plots a histogram of the specified variable as interactive
    plotly figure. Save it in html format. With the right background for the website.
    """
    fig = px.histogram(variable, x= parameter, title=title, color_discrete_sequence=color)
    fig.update_layout(paper_bgcolor="#FFF8D3") # website background color
    # save it in html in test folder
    fig.write_html(f'tests/{html_output}.html')
    fig.show()

def plot_evolution(dataframe,color,title,html_output):
    diversity_by_year = dataframe.groupby(dataframe['Movie_release_date']).apply(lambda x: pd.Series({
        'average_diversity': x['diversity'].mean(),
        'std_diversity': x['diversity'].std()
    }))

    fig = go.Figure()
    fig.update_layout(width=1000, height=600)
    fig.add_trace(go.Scatter(
        x=diversity_by_year.index,
        y=diversity_by_year.average_diversity,
        mode='lines+markers',
        name='Average Diversity',
        line=dict(color=color[0]),
        marker=dict(size=8)
    ))
    # Format x-axis to show decades
    fig.update_xaxes(
        ticktext=['1920s','1930s', '1940s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s'],
        tickvals=[1920,1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020],
        dtick=10
    )
    # Add a title and axis labels
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Diversity Score",
        legend_title="Metrics",
        template="plotly_white",
        width=1000, height=600
        # paper_bgcolor=background_color[0]
    )
    # Set the background color
    set_background_color(fig)
    # Show the interactive plot
    fig.show()
    # save it in html for the website
    fig.write_html(f'tests/{html_output}.html')


def plot_movie_release_date(df,color,title,html_output):
    """
    Plot the number of movies released per year. and save it in html format.
    """
    # Sort df by year
    df['Movie_release_date'] = df['Movie_release_date'].astype(int)
    fig = px.histogram(df['Movie_release_date'], x="Movie_release_date",color_discrete_sequence=color,title=title)
    fig.update_xaxes(title_text="Movie release date")
    fig.update_yaxes(title_text="Number of movies")
    fig.update_layout(width=1000, height=600)
    set_background_color(fig)
    fig.write_html(f'tests/{html_output}.html') # save it in html in test folder
    fig.show()
    #reset df type as int

def plot_trend_line(df,color_diversity,color_trend,title,html_output):
    """
    Plot the average diversity score per year with a trend line.
    """
    # Filter the data for the years between 1960 and 2023
    filtered_df = df[df['Movie_release_date'].between(1960, 2023)]
    
    diversity_by_year = filtered_df.groupby(filtered_df['Movie_release_date']).apply(
        lambda x: pd.Series({'average_diversity': x['diversity'].mean()})
    ).reset_index()
    # Linear regression (trend line)
    z = np.polyfit(diversity_by_year['Movie_release_date'], diversity_by_year['average_diversity'], 1)
    p = np.poly1d(z)
    
    # Create the figure
    fig = go.Figure()

    # Add line for average diversity score
    fig.add_trace(go.Scatter(
        x=diversity_by_year['Movie_release_date'],
        y=diversity_by_year['average_diversity'],
        mode='lines+markers',
        name='Average Diversity',
        line=dict(color=color_diversity[0]),
        marker=dict(size=8, color=color_diversity[0])
    ))

    # Add trend line
    fig.add_trace(go.Scatter(
        x=diversity_by_year['Movie_release_date'],
        y=p(diversity_by_year['Movie_release_date']),
        mode='lines',
        name='Trend Line',
        line=dict(color=color_trend[0], dash='dash')
    ))

    # Customize the layout
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Diversity Score",
        template="plotly_white",
        legend_title="Metrics"
    )

    # Set the background color
    set_background_color(fig)
    # Show the interactive plot
    fig.write_html(f'tests/{html_output}.html')
    fig.show()

def get_subset_box_office(df):
    """
    Get a subset of the box office data for the analysis.
    """
    # Filter the data for the years between 1960 and 2023
    subset_box_office = df.dropna(subset=['Movie_box_office_revenue'])
    return subset_box_office

def plot_interactive_bar_plot(categorie,diversity,name, color_high, color_low, title, html_output):
    """
    Plot an interactive bar plot with the specified categories and diversity scores.
    """
    # Create the figure
    fig = go.Figure()
    colors = color_palette(color_high) + color_palette(color_low)
    # Add the bar plot for high diversity
    fig.add_trace(go.Bar(
        x=categorie, y=diversity,
        name=name,
        marker=dict(color=colors)
    ))

    # Customize the layout
    fig.update_layout(
        title=title,
        barmode='group',
        xaxis_title="Categories",
        yaxis_title="Average Diversity",
        xaxis_tickangle=-30,
        showlegend=True
    )

    # Set the background color
    set_background_color(fig)
    # Show the interactive plot
    fig.write_html(f'tests/{html_output}.html')
    fig.show()


def mean_diversity(df, ratings_quantile=0.75, box_office_quantile= 0.75):
    """
    Calculate the average diversity score for success parameters:
    - Overall success
    - Nominations movies
    - Ratings
    - Box office revenue
    """

    # Define the threshold for ratings and box office revenue
    ratings_threshold = df['Ratings'].quantile(ratings_quantile)
    box_office_threshold = df['Movie_box_office_revenue'].quantile(box_office_quantile)

    # Calculate the average diversity score for different success parameters
    # Average diversity for overall success movies
    avg_diversity_overall_0 = df.loc[df['Success'] == False]['diversity'].mean()
    avg_diversity_overall_1 = df.loc[df['Success'] == True]['diversity'].mean()

    # Average diversity for film nominated / un-nominated
    avg_diversity_nominated_0 = df.loc[df['Nomination'] == False]['diversity'].mean()
    avg_diversity_nominated_1 = df.loc[df['Nomination'] == True]['diversity'].mean()
    
    # Average diversity for film with high ratings / low ratings
    diversite_ratings_0 = df.loc[df['Ratings'] <= ratings_threshold]['diversity'].mean()
    diversite_ratings_1 = df.loc[df['Ratings'] > ratings_threshold]['diversity'].mean()

    # Define the subset box office data: 
    subset_box_office = get_subset_box_office(df)

    # Average diversity for film with high box office revenue / low box office revenue
    avg_diversity_box_office_0 = subset_box_office.loc[subset_box_office['Movie_box_office_revenue'] <= box_office_threshold]['diversity'].mean()
    avg_diversity_box_office_1 = subset_box_office.loc[subset_box_office['Movie_box_office_revenue'] > box_office_threshold]['diversity'].mean()

    # # Print the results on overall success
    # print(f"Average diversity for film successful: {avg_diversity_overall_1:.4f}")
    # print(f"Average diversity for film less successful: {avg_diversity_overall_0:.4f}")
    # # Print the results on nomination
    # print(f"Average diversity for film nominated: {avg_diversity_nominated_1:.4f}")
    # print(f"Average diversity for film not nominated: {avg_diversity_nominated_0:.4f}")
    # # Print the results on ratings
    # print(f"Average diversity for film with high ratings: {diversite_ratings_1:.4f}")
    # print(f"Average diversity for film with lower ratings: {diversite_ratings_0:.4f}")
    # # Print the results on box office revenue
    # print(f"Average diversity for film with high box office revenue: {avg_diversity_box_office_1:.4f}")
    # print(f"Average diversity for film with lower box office revenue: {avg_diversity_box_office_0:.4f}")
    

    # Prepare data for interactive bar plots
    # Define the categories and average diversity scores on overall success
    categories_success = ['Successful', 'Not Successful']
    diversity_success = [avg_diversity_overall_1, avg_diversity_overall_0]

    # Define the categories and average diversity scores on nominations
    categories_nomination = ['Nominated', 'Not Nominated']
    diversity_nomination = [avg_diversity_nominated_1, avg_diversity_nominated_0]

    # Define the categories and average diversity scores on ratings
    categories_ratings = ['High Ratings', 'Low Ratings']
    diversity_ratings = [diversite_ratings_1, diversite_ratings_0]

    # define the categories and average diversity scores on box office revenue
    categories_box_office = ['High Box Office Revenue', 'Low Box Office Revenue']
    diversity_box_office = [avg_diversity_box_office_1, avg_diversity_box_office_0]

    color_high = 'beige_plus_plus'
    color_low = 'dark_red'
    # Plot the bar plots for each criteria
    # Plot overall success
    plot_interactive_bar_plot(categories_success, diversity_success,
                              'Success', color_high, color_low,
                              'Average Diversity on overall Success', 'diversity_success')
    # Plot nominations
    plot_interactive_bar_plot(categories_nomination, diversity_nomination,
                              'Nomination', color_high, color_low,
                              'Average Diversity by Nomination', 'diversity_nomination')
    
    # Plot ratings
    plot_interactive_bar_plot(categories_ratings, diversity_ratings,
                              'Ratings', color_high, color_low,
                              'Average Diversity by Ratings', 'diversity_ratings')
    # Plot box office revenue
    plot_interactive_bar_plot(categories_box_office, diversity_box_office,
                                'Movie_box_office_revenue', color_high, color_low,
                                'Average Diversity by Box Office Revenue', 'diversity_box_office')

    # # Create the interactive bar plots
    # fig = go.Figure()

    # # Plot for Nomination
    # fig.add_trace(go.Bar(
    #     x=categories_nomination, y=diversity_nomination,
    #     name='Nomination',
    #     marker=dict(color=['#B77526', '#A01812'])
    # ))

    # # Plot for Ratings
    # fig.add_trace(go.Bar(
    #     x=categories_ratings, y=diversity_ratings,
    #     name='Ratings',
    #     marker=dict(color=['#B77526', '#A01812']),
    
    # ))

    # # Plot for Success
    # fig.add_trace(go.Bar(
    #     x=categories_success, y=diversity_success,
    #     name='Success',
    #     marker=dict(color=['#B77526', '#A01812']),
    
    # ))

    # # Update layout for better appearance
    # fig.update_layout(
    #     title="Diversity Analysis by Different Factors",
    #     barmode='group',
    #     xaxis_title="Categories",
    #     yaxis_title="Average Diversity",
    #     xaxis_tickangle=-30,
    #     showlegend=True
    # )

    # # Show the interactive plot
    # fig.show()