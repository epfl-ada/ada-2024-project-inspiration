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
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

background_color = ["#FFF8D3"] # website background color

# Set figure size for beautiful-jekyll compatibility
FIGURE_WIDTH = 800   # pixels
FIGURE_HEIGHT = 600  # pixels

def set_figsize(fig,width=FIGURE_WIDTH, height=FIGURE_HEIGHT):
    """Set the figure size for matplotlib plots."""
    fig.update_layout(width=width, height=height)

def color_palette(color):
    """take a color name as an argument and return the color in hex format
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
    """
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
    set_figsize(fig)
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
        width=800, height=600
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
    set_figsize(fig,800)
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
    set_figsize(fig)
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

def plot_interactive_bar_plot(categorie,diversity,name, std_success,color_high, color_low, title, html_output):
    """
    Plot an interactive bar plot with the specified categories and diversity scores.
    """
    # Create the figure
    fig = go.Figure()
    set_figsize(fig)
    colors = color_palette(color_high) + color_palette(color_low)
    # Add the bar plot for high diversity
    fig.add_trace(go.Bar(
        x=categorie, y=diversity,
        error_y=dict(
        type='data',
        array=std_success,
        visible=True),
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

    # Prepare data for interactive bar plots
    # Define the categories and average diversity scores on overall success
    categories_success = ['Successful', 'Not Successful']
    diversity_success = [avg_diversity_overall_1, avg_diversity_overall_0]

    # define the categories and average diversity scores on box office revenue
    categories_box_office = ['High Box Office Revenue', 'Low Box Office Revenue']
    diversity_box_office = [avg_diversity_box_office_1, avg_diversity_box_office_0]

    # Define the categories and average diversity scores on ratings
    categories_ratings = ['High Ratings', 'Low Ratings']
    diversity_ratings = [diversite_ratings_1, diversite_ratings_0]

    # Define the categories and average diversity scores on nominations
    categories_nomination = ['Nominated', 'Not Nominated']
    diversity_nomination = [avg_diversity_nominated_1, avg_diversity_nominated_0]

    # Store the results in a table
    mean_table = [categories_success, diversity_success, 
                  categories_box_office, diversity_box_office,
                  categories_ratings, diversity_ratings,
                  categories_nomination, diversity_nomination
                  ]
    return mean_table

def std_diversity(df, ratings_quantile=0.75, box_office_quantile= 0.75):
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
    std_diversity_overall_0 = df.loc[df['Success'] == False]['diversity'].std()
    std_diversity_overall_1 = df.loc[df['Success'] == True]['diversity'].std()

    # Average diversity for film nominated / un-nominated
    std_diversity_nominated_0 = df.loc[df['Nomination'] == False]['diversity'].std()
    std_diversity_nominated_1 = df.loc[df['Nomination'] == True]['diversity'].std()
    
    # Average diversity for film with high ratings / low ratings
    diversite_ratings_0 = df.loc[df['Ratings'] <= ratings_threshold]['diversity'].std()
    diversite_ratings_1 = df.loc[df['Ratings'] > ratings_threshold]['diversity'].std()

    # Define the subset box office data: 
    subset_box_office = get_subset_box_office(df)

    # Average diversity for film with high box office revenue / low box office revenue
    std_diversity_box_office_0 = subset_box_office.loc[subset_box_office['Movie_box_office_revenue'] <= box_office_threshold]['diversity'].std()
    std_diversity_box_office_1 = subset_box_office.loc[subset_box_office['Movie_box_office_revenue'] > box_office_threshold]['diversity'].std()   

    # Prepare data for interactive bar plots
    # Define the categories and average diversity scores on overall success
    categories_success = ['Successful', 'Not Successful']
    diversity_success = [std_diversity_overall_1, std_diversity_overall_0]

    # define the categories and average diversity scores on box office revenue
    categories_box_office = ['High Box Office Revenue', 'Low Box Office Revenue']
    diversity_box_office = [std_diversity_box_office_1, std_diversity_box_office_0]

    # Define the categories and average diversity scores on ratings
    categories_ratings = ['High Ratings', 'Low Ratings']
    diversity_ratings = [diversite_ratings_1, diversite_ratings_0]

    # Define the categories and average diversity scores on nominations
    categories_nomination = ['Nominated', 'Not Nominated']
    diversity_nomination = [std_diversity_nominated_1, std_diversity_nominated_0]

    # Store the results in a table
    std_table = [categories_success, diversity_success, 
                  categories_box_office, diversity_box_office,
                  categories_ratings, diversity_ratings,
                  categories_nomination, diversity_nomination
                  ]
    return std_table

def get_thresholds(df, ratings_quantile, box_office_quantile):
    """
    Get the thresholds for ratings and box office revenue.
    """

    # Define the threshold for ratings and box office revenue
    ratings_threshold = df['Ratings'].quantile(ratings_quantile)
    box_office_threshold = df['Movie_box_office_revenue'].quantile(box_office_quantile)

    return ratings_threshold, box_office_threshold

# def store_t_test(t_test, metric):
#     """
#     Store the t-test results in a DataFrame and save it to HTML.
#     """
#     # Create a DataFrame for the t-test results with a single row
#     styled_df = pd.DataFrame({
#         'Metric': [metric],
#         'Statistic': [t_test.statistic],
#         'P-value': [t_test.pvalue]
#     }).set_index('Metric')

#     styled_df.to_html(f'tests/t_test_{metric}.html')
#     return styled_df

def store_t_test(t_test, metric):
    """
    Store the t-test results in a DataFrame and save it to HTML.
    """
    # Create a DataFrame with the desired structure
    styled_df = pd.DataFrame({
        'Metric': [' ', ' '],  # Empty rows for Metric
        'Statistic': [t_test.statistic, ' '],  # Statistic on the first row
        'P-value': [t_test.pvalue, ' ']        # P-value on the second row
    })
    
    # Add the metric as the index for the first row
    styled_df.iloc[0, 0] = metric

    # Save the DataFrame to an HTML file
    styled_df.to_html(f'tests/t_test_{metric}.html', index=False)
    return styled_df

def get_t_tests(df, ratings_quantile=0.75, box_office_quantile=0.75):
    """
    Perform t-tests to compare diversity scores between successful and unsuccessful movies.
    """
    ratings_threshold, box_office_threshold = get_thresholds(df, ratings_quantile, box_office_quantile)
    # Perform t-tests to compare diversity scores between successful and unsuccessful movies
    t_test_sucess=stats.ttest_ind(df.loc[df['Success'] == True]['diversity'], df.loc[df['Success'] == False]['diversity'])
    t_test_nomination=stats.ttest_ind(df.loc[df['Nomination'] == True]['diversity'], df.loc[df['Nomination'] == False]['diversity'])
    t_test_box=stats.ttest_ind(df.loc[df['Movie_box_office_revenue'] > box_office_threshold]['diversity'], df.loc[df['Movie_box_office_revenue'] <= box_office_threshold]['diversity'])
    t_test_ratings=stats.ttest_ind(df.loc[df['Ratings'] > ratings_threshold ]['diversity'], df.loc[df['Ratings'] <= ratings_threshold ]['diversity'])

    # Store the t-test results in a DataFrame and save it to HTML
    t_test_sucess = store_t_test(t_test_sucess, 'Overall_success')
    t_test_nomination = store_t_test(t_test_nomination, 'Nomination')
    t_test_ratings = store_t_test(t_test_ratings, 'Ratings')
    t_test_box = store_t_test(t_test_box, 'Box_office_revenue')
    return t_test_sucess, t_test_nomination, t_test_ratings, t_test_box


def regression(df, features, target):
    # Split features and target
    X = df[features]
    y = df[target]
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

    # we add constant column
    X_train_scaled = sm.add_constant(X_train_scaled)
        
    # Create and train the model
    model = sm.OLS(y_train, X_train_scaled).fit()
    
    # Scale and prepare test data
    X_test_scaled = scaler.transform(X_test.values)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    X_test_scaled = sm.add_constant(X_test_scaled)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    # Create results dataframe
    results_df = pd.DataFrame({
        #'X_test': X_test, # .values.tolist(),
        'y_test': y_test,
        'y_pred': y_pred
    })
    results_df = pd.concat([results_df, X_test], axis=1)
    if model.rsquared < 0.1:
        print('WARNING: R-squared is below 0.1')
        print(f'R-squared: {model.rsquared:.4f}')
        print("As the R-squared is below 0.1, the model is not good enough to fit the data.")
    
    return results_df, model

def get_thresholds_diversity(df, diversity_quantile):
    diversity_threshold = df['diversity'].quantile(diversity_quantile)
    return diversity_threshold

def count_countries(countries_str):
    countries = countries_str.split(',')  
    return len(countries)

def count_languages(languages_str):
    languages = languages_str.split(',')  
    return len(languages)

def get_similarity(propensity_score1, propensity_score2):
    '''Calculate similarity for instances with given propensity scores'''
    return 1-np.abs(propensity_score1-propensity_score2)

def propensity_score(df):
    #let's reduce the dataset to get only movies with revenue that are no nan. 
    dataframe=get_subset_box_office(df)

    #get the Q3 threshold for diversity. 
    diversity_threshold=get_thresholds_diversity(dataframe,diversity_quantile=0.75)

    #add three column treat, number of countries and number of languages. 
    dataframe['Number_of_countries'] = dataframe['Movie_countries'].apply(count_countries)
    dataframe['Number_of_languages'] = dataframe['Movie_languages'].apply(count_languages)
    dataframe['treat']=(dataframe['diversity'] > diversity_threshold).astype(int)

    # let's standardize the continuous features. 
    dataframe['Movie_release_date'] = (dataframe['Movie_release_date'] - dataframe['Movie_release_date'].mean())/dataframe['Movie_release_date'].std()
    dataframe['Number_of_countries'] = (dataframe['Number_of_countries'] - dataframe['Number_of_countries'].mean())/dataframe['Number_of_countries'].std()
    dataframe['Number_of_languages'] = (dataframe['Number_of_languages'] - dataframe['Number_of_languages'].mean())/dataframe['Number_of_languages'].std()

    mod = smf.logit(formula= 'treat ~  Movie_release_date + Number_of_countries + Number_of_languages' , data=dataframe)
    res = mod.fit()

    dataframe['Propensity_score'] = res.predict()

    dataframe=dataframe.sample(n=500,random_state=42)
    
   # Treatment is diversity. 
    treatment_df = dataframe[dataframe['treat'] == 1]
    control_df = dataframe[dataframe['treat'] == 0]

    # Create an empty undirected graph
    G = nx.Graph()

    # Loop through all the pairs of instances
    for control_id, control_row in control_df.iterrows():
        for treatment_id, treatment_row in treatment_df.iterrows():

        # Calculate the similarity
            similarity = get_similarity(control_row['Propensity_score'],treatment_row['Propensity_score'])

        # Add an edge between the two instances weighted by the similarity between them
            G.add_weighted_edges_from([(control_id, treatment_id, similarity)])

# Generate and return the maximum weight matching on the generated graph
    matching = nx.max_weight_matching(G)
    matched_indices = [node for edge in matching for node in edge]
    balanced_df_1 = dataframe.loc[matched_indices]
    return balanced_df_1, len(matching)

def get_ATE(dataframe,number):
    treated = dataframe.loc[dataframe['treat'] == 1]
    control = dataframe.loc[dataframe['treat'] == 0]
    y_treat= treated['Movie_box_office_revenue'].sum()
    y_control= control['Movie_box_office_revenue'].sum()
    ATE= (1/number)* (y_treat-y_control)
    return print(f"Average Treatment Effect (ATE) : {ATE}")

def plot_propensity(dataframe,number, color_group_1, color_group_2, title, html_output):
    """
    Plot the propensity score distribution for the treated and control groups.
    """
    treated = dataframe.loc[dataframe['treat'] == 1]
    control = dataframe.loc[dataframe['treat'] == 0]
    # Create a histogram for treated
    treated_hist = go.Histogram(
        x=treated['Movie_box_office_revenue'],
        name='Treated',
        marker=dict(color=color_group_1[0]),
        opacity=0.6
    )

    # Create a histogram for control
    control_hist = go.Histogram(
        x=control['Movie_box_office_revenue'],
        name='Control',
        marker=dict(color=color_group_2[0]),
        opacity=0.4
    )
     # Create a figure and add both histograms
    fig = go.Figure(data=[treated_hist, control_hist])
    set_background_color(fig)
    set_figsize(fig)
    fig.update_layout(title=title,
        xaxis_title='Box office revenue',
        yaxis_title='Movie count',
        legend_title = 'Group',
        barmode='overlay',
        plot_bgcolor='white')
    # Save the plot as an HTML file in the test folder
    fig.write_html(f'tests/{html_output}.html')

    # Display the figure
    fig.show()