# THE SUCCESS OF INCLUSIVE CINEMA
This is a template repo for your project to help you organise and document your code better. 
Please use this structure for your project and document the installation, usage and structure as below.


## How to use the library
For our project pipeline, we used the given project template. We added 2 subdirectories to data : \raw_data with the uncleaned datasets and \processed_data with the 
cleaned data sets

all the dataset used for our project are available trhough this google drive link : 
https://drive.google.com/drive/folders/15jMzHA16hZhBR0086Y3cT6uLr1BEbrLo?usp=sharing


## Link to the website
https://flore-mueth.github.io/ADA_P3-jekyll/

## Project Structure

The directory structure of our project is as follows:
```
├── data                        <- Project data files
│   ├── raw_data                <- Data unprocessed directory
│   ├── processed_data          <- Processed data directory     
│       ├── clean_dataset.csv                   <- cleaned dataset with all the information about the movies 
│       ├── success_and_diversity.csv           <- dataset used for main analysis with diversity and success
| 
├── src                         <- Source code
│   ├── data                            <- Data directory
│       ├── cleaning_data.py                    <- file for cleaning our datasets
│       ├── data_exploration.py                 <- file with our main analysis
│       ├── data_visualisation.py               <- data visualization file
│       ├── diversity.py                        <- file for defining the diversity of a movie
│       ├── success.py                          <- file for defining the success of a movie
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│       ├── load_awards_data.py                 <- file for loading awards data in \processed_data
│
├── tests                       <- plots for the data visualisation on website saved in .html
│
├── results.ipynb               <- a well-structured notebook showing the results
├── preliminary_results_P2.ipynb<- Notebook with preliminary results from milestone 2
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```
# Detailed project proposal from milestone 2

**Is diversity a factor of success for films?** 

## Abstract:

The relationship between diversity in film casts and box office success is increasingly relevant in light of new industry standards and social awareness. However, the degree to which the cast diversity directly influences a film’s success remains unclear. In this work, we aim to explore whether films featuring diverse casts perform better financially and critically, examining data on film diversity, box office earnings, and awards from the perspective of the Academy’s 2024 Oscars diversity and inclusion standards. By doing so, we seek to gain insight into how diversity impacts industry success indicators, offering a data-driven narrative on whether representation positively correlates with audience and critical reception. This analysis could contribute to discussions on diversity's value not only as a social objective but also as a potential factor of commercial success, shaping a clearer understanding for filmmakers and industry stakeholders.

## 1) Research questions:

**Does diversity in the film cast correlate with a film’s success?**

*  **Financial Success**  
  * How does cast diversity affect box office performance?  
  * What is a high box revenue?  
  * How does the impact of diversity on box office performance vary across different genres and budget levels?  
*  **Critical and Audience Reception**  
  * How does cast diversity influence critical reception and award nomination?  
  * Do films with diverse casts receive higher or lower ratings from audiences and critics?  
  * What is a high rate? 

## 2) Methods:

### **TASK 1: Getting to know the data**

**Clean the data:**   
**1)** remove missing value and non usable one :   
- character & movie metadata rename title   
- plot missing value and remaining column  
- keep only the column of interest : GENERAL : wiki_ID, release date, movie name, actor ethnicity, movie country. FOR SUCCESS ANALYSIS : + box office, note du public, oscar nomination  
**2)** clean the value, we want all the column written in the format without value that are not understable or empty (like {})  
**3)** Create a final dataframe with all the informations about character and movie → merging on wiki\_id or on movie name in function of the step or what we want

### **TASK 2: Data enrichment** 

To conduct our analysis, we decided to import other databases: 

- **fb_wiki_mapping.tsv**: The additional dataset links freebase_id values to wikidata_id and label, enhancing the project by enabling connections to Wikidata and adding descriptive labels. This mapping enriches data from the primary datasets, providing access to additional metadata and improving interpretability.  In the code, we use a function that searches the TSV file for matching freebase_id entries, returning the wikidata_id and label if found, or None if not.  
- Two additional datasets available on [IMDb's website](https://datasets.imdbws.com/). This dataset provides us with the movie name and movies’ ratings, users ratings on a scale from 0 to 10.  We worked with 1\) **title.basics.tsv**: This dataset allows us to retrieve the title of the film associated with a unique identifier. 2\) **title.ratings.tsv**: This dataset allows us to retrieve IMDb ratings of films associated with a unique identifier. These dataset will enable us to analyze film ratings and understand whether cast diversity influences user scores.

Database from worldwide acknowledged awards, 

- Oscar award for best pictures nominees list from Wikipedia.   
- Golden Globes award for best pictures nominees list from Wikipedia  
- Filmfare awards for best pictures nominees list from Wikipedia, from Bollywood industry  
- Golden palm winners list from Wikipedia  
- Asian films awards  for best pictures nominees list from Wikipedia  
- Cesar nominees

All these datasets were cleaned and filtered before merging them to our original data.  

### **TASK 3: Initial analysis**

#### 1) **how can we define diversity and group ethnicities**

- Define Major Groups: Use ChatGPT to determine the optimal number of major ethnic groups and create a clustering approach for grouping ethnicities accordingly.

- First Analysis (Basic Approach): Calculate diversity as the number of distinct ethnicities in a film, normalized by the total number of characters. This results in a percentage representing diversity.

- Second Analysis (Refined Approach): Assess the distribution of ethnicities within the film. For example, consider if ethnic representation is balanced (e.g., 50-50, 1/3-1/3-1/3). This balance score can then be used as a coefficient to adjust and refine the diversity percentage calculated in the first analysis.

#### 2) **Definition of a successful movie:** 

We define a successful film based on three criteria: high box office revenue, high user ratings, or award nominations (including the Oscars, Golden Globes, Palme d'Or, Filmfare Awards, and Asian Film Awards).

To ensure a sufficient sample size of successful films, we think of setting thresholds for each criterion:

-High Box Office Revenue: We will specify the threshold for box office earnings to classify a film as financially successful.

-High User Ratings: Films with an average user rating of 7/10 or higher are considered highly rated.

-Award Nominations: Rather than limiting our selection to award winners, which could be too restrictive, we include all films nominated for Oscars and Golden Globes, among other prominent awards.

Conclusion:  we have shown that our project is feasible 

### **TASK 4: Analysis and plan to answer research questions (TBD in Milestone 3)**

This section forms the core of our analysis, aiming to assess the correlation between cast diversity and film success. Specifically, we want to determine whether diversity is more prevalent in successful films or if no discernible pattern exists. We will analyze films categorized into two groups—successful and less successful—based on established success criteria.

#### 1) **Comparing Diversity Across Success Levels**

-Calculate the average diversity score for both successful and less successful films: average diversity in successful films and average diversity in less successful films. Examine the difference in diversity averages between the two groups.

-Significance Testing: Conduct a two-sided t-test (using stats.ttest_ind) to evaluate if the diversity difference between successful and less successful films is statistically significant. The null hypothesis is that both groups have identical average diversity scores.

#### 2) **Assessing Correlation between Diversity and Success**

-Correlation Analysis: Evaluate the dependency between diversity and film success using: Pearson correlation (stats.pearson) for linear associations.

-Spearman correlation (stats.spearman) 

-Regression Analysis: Investigate the association type (e.g., weak, negative, positive) through linear regression.

#### 3) **Measuring Uncertainty**

Quantify the uncertainty in our findings to strengthen the reliability of our results (confidence intervals, standards errors..)

By conducting these analyses, we aim to uncover any meaningful relationship between diversity and success in films, providing insight into whether increased diversity contributes to or aligns with box office and critical achievements.


## **Proposed timeline:**

15.11.2024  Data Handling and Preprocessing & Initial Exploratory Data Analysis  
29.11.2024 H2 Deadline/ Task 3 has to be finished.   
13.12.2024 Task 4 has to be finished.   
20.12.2024 tout mettre en commun/ désigner le site internet/ Milestone 3 deadline

### **Organization within the team:** 

A list of internal milestones up until project Milestone P3.  
Mathilde:Statistic analysis   
Berend:Ethnicity: Make a function to define ethnicity value of a film  
Albane:Statistic analysis   
Flore: Statistic analysis   
Francesco: Ethnicity: better definition of ethnic groups.

# Milestone 3

## Clean the data
### On the given dataset
From the given datasets, we have cleaned the data by removing missing values and non-usable characters. We kept only the column of interest :
- GENERAL : Movie name, release date, ratings , Wikipedia_ID, Box office revenue, Movie countries, Movies languages, actor ethnicity
### additional datasets
We have also cleaned the additional datasets we imported. We kept only the columns of interest and removed the missing values. We also merged the datasets to our original data. After defining our diversity and success criteria. 


## Treatment of the diversity data and establishing the coefficient

### 1) Treating the ethnicities

When we first have a look at the ethnicities, we can see that there are a total of more than 350 different ethnicities, some of them still very similar (e.g. 'Austrian American' and 'Austrian Canadian' etc). We want to first simplify this ethnicity criterion before defining diversity. If we didn’t sort the ethnicities, a film with a cast of a German, Austrian and Swiss would be considered very diverse. This is however not what we want to consider diverse. It is for this reason that the ethnicities were first grouped into larger ethnic groups. This was done with the help of a LLM, with checks and corrections done by hand. Doing this by hand was still possible thanks to the manageable number of ethnicities and the LLM doing the most time-consuming part.

### 2) Defining diversity

Once the ethnicities have been sorted into larger groups (16), we can start defining diversity. The focus being on the diversity of the cast and not the representation of minority groups, country of production of the movie doesn’t have to be taken into account. The first and easiest way to calculate diversity would be dividing the number of ethnicities over the number of actors. However, for a film with 9 actors and 3 ethnicities, this definition would give the same diversity score for a distribution (3,3,3) as for (1,1,7). Calculating an entropy could therefore complete the previous definition. The basic entropy formula is: 

S \= Σpi·ln(pi) , pi being the fraction an ethnicity represents in the movie .

**One modification was made**. Indeed, if we have all actors from 1 ethnicity, we get an entropy of 0, but it is preferable to avoid the value of 0 since we will multiply the entropy with the other definition. We therefore have added 1 to the entropy. This entropy penalises the movies with smaller numbers of actors, which is why we have multiplied entropy with the first definition to establish our final diversity coefficient

### 3) Further comments

Firstly, the data set gives us many movies with different numbers of actors and all the movies with 1 actor cannot be considered for a diversity calculation. If we wanted to further complete the analysis we could consider whether an actor is from a minority group.

Secondly, the diversity coefficient is based on the ethnic groups established previously. Changing the characteristics of the ethnic groups, such as their size, their number or their content will change the diversity factor.

## Success Definition
Success is defined based on several parameters:
- **Nominations**: Whether the movie has received any nominations.
- **Ratings**: The ratings of the movie, with a threshold defined by the 75th percentile.
- **Box Office Revenue**: The box office revenue of the movie, with a threshold defined by the 75th percentile.

We created a global success score by considering a movie successful when it meets at least one of the three criteria. This score is used to compare the success of movies with different criteria. We obtain a pourcentage of 25 % of successful movies. Which is a good balance between the two groups.

## Design of Plot Functions
The plot functions are designed to visualize various aspects of the data, such as the distribution of diversity scores, the evolution of diversity over time, and the relationship between diversity and success parameters. The functions use Plotly for interactive visualizations and are customized to match the website's color scheme.

## Main analysis
### 1) **Comparing Diversity Across Success**
We calculated the average diversity score for successful and less successful films and found that successful films tend to have higher diversity scores. The difference in diversity averages between the two groups is statistically significant, indicating that diversity may be a factor in a film's success.

### 2) **Assessing Correlation between Diversity and Success**
We evaluated the dependency between diversity and film success using Pearson and Spearman correlation coefficients. The results suggest a positive correlation between diversity and success, with higher diversity scores associated with higher success scores. Linear regression analysis further supports this finding, indicating a positive relationship between diversity and success.

### 3) **Measuring Uncertainty**
We quantified the uncertainty in our findings by calculating confidence intervals and standard errors. The results show that our findings are not that robust and reliable.

## Interpretation of Results
The results are interpreted in the context of the research questions, focusing on the relationship between diversity and success in films. The analysis considers the impact of diversity on box office revenue, ratings, and award nominations, providing insights into the potential benefits of diversity in the film industry. We could not find a strong correlation between diversity and success, but the results suggest that diversity may contribute positively to a film's critical reception and audience ratings.

## Organisation within the team

- Mathilde: Statistical analysis and interpretation
- Albane: Statistical analysis and interpretation
- Flore: website design and implementation, data cleaning
- Francesco: Project structuration, data visualization 
- Berend : definition of diversity and diversity score