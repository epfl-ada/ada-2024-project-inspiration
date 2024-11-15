# Your project name
This is a template repo for your project to help you organise and document your code better. 
Please use this structure for your project and document the installation, usage and structure as below.

## Quickstart

```bash
# clone project
git clone <https://github.com/epfl-ada/ada-2024-project-inspiration>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>


# install requirements
pip install -r pip_requirements.txt
```



### How to use the library
For our project pipeline, we used the given project template. We added 2 subdirectories to data : \raw_data with the uncleaned datasets and \processed_data with the 
cleaned data sets



## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│   ├── raw_data                <- Data unprocessed directory
│   ├── processed_data          <- Processed data directory      
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│       ├── load_awards_data.py                 <- file for loading awards data in \processed_data
│       ├── data_visualisation.py               <- data visualization file called by results.ipynb
│       ├── load_awards_data.py                 <- file for cleaning our datasets
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```
### Detailed project proposal

**Is diversity a factor of success for films?** 

# Abstract:

The relationship between diversity in film casts and box office success is increasingly relevant in light of new industry standards and social awareness. However, the degree to which the cast diversity directly influences a film’s success remains unclear. In this work, we aim to explore whether films featuring diverse casts perform better financially and critically, examining data on film diversity, box office earnings, and awards from the perspective of the Academy’s 2024 Oscars diversity and inclusion standards. By doing so, we seek to gain insight into how diversity impacts industry success indicators, offering a data-driven narrative on whether representation positively correlates with audience and critical reception. This analysis could contribute to discussions on diversity's value not only as a social objective but also as a potential factor of commercial success, shaping a clearer understanding for filmmakers and industry stakeholders.

1) # Research questions:

**Does diversity in the film cast correlate with a film’s success?**

*  **Financial Success**  
  * How does cast diversity affect box office performance?  
  * What is a high box revenue?  
  * How does the impact of diversity on box office performance vary across different genres and budget levels?  
*  **Critical and Audience Reception**  
  * How does cast diversity influence critical reception and award nomination?  
  * Do films with diverse casts receive higher or lower ratings from audiences and critics?  
  * What is a high rate? 

2) # Methods:

## **TASK 1: Getting to know the data**

**Clean the data:**   
**1\)** remove missing value and non usable one :   
\- character et movie metadata rename title   
\- plot missing value and remaining column  
\- keep only the column of interest : GENERAL : wiki\_ID, release date, movie name, actor ethnicity, movie country. FOR SUCCESS ANALYSIS : \+ box office, note du public, oscar nomination  
**2\)** clean the value, we want all the column written in the format without value that are not understable or empty (like {})  
**3\)** Create a final dataframe with all the informations about character and movie → merging on wiki\_id or on movie name in function of the step or what we want

## **TASK 2: Data enrichment** 

To conduct our analysis, we decided to import other databases: 

- **fb\_wiki\_mapping.tsv**: The additional dataset links freebase\_id values to wikidata\_id and label, enhancing the project by enabling connections to Wikidata and adding descriptive labels. This mapping enriches data from the primary datasets, providing access to additional metadata and improving interpretability.  In the code, we use a function that searches the TSV file for matching freebase\_id entries, returning the wikidata\_id and label if found, or None if not.  
- Two additional datasets available on [IMDb's website](https://datasets.imdbws.com/). This dataset provides us with the movie name and movies’ ratings, users ratings on a scale from 0 to 10\.  We worked with 1\) **title.basics.tsv**: This dataset allows us to retrieve the title of the film associated with a unique identifier. 2\) **title.ratings.tsv**: This dataset allows us to retrieve IMDb ratings of films associated with a unique identifier. These dataset will enable us to analyze film ratings and understand whether cast diversity influences user scores.

Database from worldwide acknowledged awards, 

- Oscar award for best pictures nominees list from Wikipedia.   
- Golden Globes award for best pictures nominees list from Wikipedia  
- Filmfare awards for best pictures nominees list from Wikipedia, from Bollywood industry  
- Golden palm winners list from Wikipedia  
- Asian films awards  for best pictures nominees list from Wikipedia  
- Cesar nominees

All these datasets were cleaned and filtered before merging them to our original data.  

## **TASK 3: Initial analysis**

1) **how can we define diversity and group ethnicities**

\-Define Major Groups: Use ChatGPT to determine the optimal number of major ethnic groups and create a clustering approach for grouping ethnicities accordingly.

\-First Analysis (Basic Approach): Calculate diversity as the number of distinct ethnicities in a film, normalized by the total number of characters. This results in a percentage representing diversity.

\-Second Analysis (Refined Approach): Assess the distribution of ethnicities within the film. For example, consider if ethnic representation is balanced (e.g., 50-50, 1/3-1/3-1/3). This balance score can then be used as a coefficient to adjust and refine the diversity percentage calculated in the first analysis.

2) **Definition of a successful movie:** 

We define a successful film based on three criteria: high box office revenue, high user ratings, or award nominations (including the Oscars, Golden Globes, Palme d'Or, Filmfare Awards, and Asian Film Awards).

To ensure a sufficient sample size of successful films, we think of setting thresholds for each criterion:

\-High Box Office Revenue: We will specify the threshold for box office earnings to classify a film as financially successful.

\-High User Ratings: Films with an average user rating of 7/10 or higher are considered highly rated.

\-Award Nominations: Rather than limiting our selection to award winners, which could be too restrictive, we include all films nominated for Oscars and Golden Globes, among other prominent awards.

Conclusion:  we have shown that our project is feasible 

## **TASK 4: Analysis and plan to answer research questions (TBD in Milestone 3\)**

This section forms the core of our analysis, aiming to assess the correlation between cast diversity and film success. Specifically, we want to determine whether diversity is more prevalent in successful films or if no discernible pattern exists. We will analyze films categorized into two groups—successful and less successful—based on established success criteria.

1) **Comparing Diversity Across Success Levels**

\-Calculate the average diversity score for both successful and less successful films: average diversity in successful films and average diversity in less successful films. Examine the difference in diversity averages between the two groups.

\-Significance Testing: Conduct a two-sided t-test (using stats.ttest\_ind) to evaluate if the diversity difference between successful and less successful films is statistically significant. The null hypothesis is that both groups have identical average diversity scores.

2) **Assessing Correlation between Diversity and Success**

\-Correlation Analysis: Evaluate the dependency between diversity and film success using: Pearson correlation (stats.pearson) for linear associations.

\-Spearman correlation (stats.spearman) 

\-Regression Analysis: Investigate the association type (e.g., weak, negative, positive) through linear regression.

3) **Measuring Uncertainty**

Quantify the uncertainty in our findings to strengthen the reliability of our results (confidence intervals, standards errors..)

By conducting these analyses, we aim to uncover any meaningful relationship between diversity and success in films, providing insight into whether increased diversity contributes to or aligns with box office and critical achievements.

**Proposed timeline:**

15.11.2024  Data Handling and Preprocessing & Initial Exploratory Data Analysis  
29.11.2024 H2 Deadline/ Task 3 has to be finished.   
13.12.2024 Task 4 has to be finished.   
20.12.2024 tout mettre en commun/ désigner le site internet/ Milestone 3 deadline

**Organization within the team:** 

A list of internal milestones up until project Milestone P3.  
Mathilde:Statistic analysis   
Berend:Ethnicity: Make a function to define ethnicity value of a film  
Albane:Statistic analysis   
Flore: Statistic analysis   
Francesco: Ethnicity: better definition of ethnic groups. and find a punchy project name

**Question for TAs,** 

- We have many very specific ethnicities and we want to have more general ethnicities. This is now done manually, but ideally we would like to use something like openai to automate this. Is this possible? And then do we have the ‘license’ or do we have a quota of questions asked, since we encountered some problems when trying to call openai directly from the script (it said the question quota was exhausted).

- What percentage of successful films should we target as a threshold to ensure meaningful outputs in our analysis?