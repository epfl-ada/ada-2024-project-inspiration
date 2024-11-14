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
Tell us how the code is arranged, any explanations goes here.



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
