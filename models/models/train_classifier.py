import sys
from sqlalchemy import create_engine
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report,confusion_matrix, precision_score,\
recall_score,accuracy_score,  f1_score,  make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk


from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


import pickle


def load_data(database_filepath):
    # load daata from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM Disaster_cleaned", engine)
    
    
    # Drop uneeded columns and missing values
    df = df.dropna()
    X = df["message"]
    Y = df.drop("message",1)
    Y = Y.drop("id",1)
    Y = Y.drop("genre",1)
    Y = Y.drop("original",1)
    category_names = Y.columns
    
    return X,Y,category_names

    pass


def tokenize(text):
    # Intialize tokenizer and lemmatizer
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # A loop to lemmatize lower strip and append the text
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    pass





def build_model():
    
    
    #initialize the pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier(n_estimators=1000, random_state=0)))
])
    
    
    
    # initialize the parameters
    parameters = [
    {
        #'clf__estimator__max_leaf_nodes': [50, 100, 200],
        #'clf__estimator__min_samples_split': [2, 3, 4],
    }
]
    # build the model with grid search
    cv = GridSearchCV(pipeline, param_grid=parameters,
                 #scoring=average_accuracy_score, 
                verbose=10, 
                return_train_score=True 
                 )
    return cv
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    """
    INPUT:
        model- The model to evaluate 
        X_test -  The data 
        Y_test -  Labels
        category_names 
    OUTPUT: 
        NONE
    """
    metrics_list_all=[]
    for col in range(y_test.shape[1]):
        
        accuracy = accuracy_score(y_test.iloc[:,col], y_pred[:,col])
        precision=precision_score(y_test.iloc[:,col], y_pred[:,col],average='micro')
        recall = recall_score(y_test.iloc[:,col], y_pred[:,col],average='micro')
        f_1 = f1_score(y_test.iloc[:,col], y_pred[:,col],average='micro')
        metrics_list=[accuracy,precision,recall,f_1]
        metrics_list_all.append(metrics_list)
        metrics_df=pd.DataFrame(metrics_list_all,index=category_names,columns=["Accuracy","Precision","Recall","F_1"])
        print(metrics_df)


    
    
    
    pass


def save_model(model, model_filepath):
    """
    INPUT:
        model 
        model_filepath - Where will the model will be saved
    OUTPUT: 
        NONE
    """    
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()