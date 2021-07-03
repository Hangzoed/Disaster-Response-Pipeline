import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    INPUT:
        messages_filepath - Path for the CSV that stores messages 
        categories_filepath - Path for the CSV that stores Categories
   OUTPUT:
        df - Panda dataframe - A data frame that cotains merged categories and messages
   """
    
    messages = pd.read_csv(messages_filepath)
    categories_filepath = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories_filepath, on=['id'])
    return df
    pass


def clean_data(df):
    """
    INPUT:
         df - Panda DataFrame - A data frame that contains the data
    OUTPUT:
         df - Panda DataFrame - A Cleaned Panda Data frame
    """
    #split categories into a data frame and take the first row
    cat = df.categories.str.split(';', expand=True)
    row = cat.iloc[0]

    rew=row.unique()
    
   # Fix columns name
    f = []
    for x in rew:
        r = x[:-2]
        f.append(r)
        

    
    category_colnames = pd.Series(f)
    cat.columns = category_colnames
    for column in cat:
        cat[column] = cat[column].str.strip().str[-1]
    
    # convert column from string to numeric
        cat[column] = cat[column].astype('int64')
    
    
    # concating the categories column with df and dropping unnesscary values
    df = df.drop(['categories'], axis = 1)

    df = pd.concat([df, cat], axis=1 )
    
    
    df = df.drop_duplicates()
    
    df.dropna(how='any')
    return df

    pass


def save_data(df, database_filename):
    """
    INPUT:
         df - A panda data frame that holds the data
         database_filename - Name of the SQL database that will hold the data
   OUTPUT:
        NONE
   """
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Disaster_cleaned' , engine, if_exists='replace', index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()