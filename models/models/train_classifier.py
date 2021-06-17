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
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    pass
def avg_accuracy_score(y_true, y_pred):
    """
        Assumes that the numpy arrays `y_true` and `y_pred` ararys 
        are of the same shape and returns the average of the 
        accuracy score computed columnwise. 

        y_true - Numpy array - An (m x n) matrix 
        y_pred - Numpy array - An (m x n) matrix 

        avg_accuracy - Numpy float64 object - Average of accuracy score
        """

    # initialise an empty list
    accuracy_results = []

    # for each column index in either y_true or y_pred
    for idx in range(y_true.shape[-1]):
        # Get the accuracy score of the idx-th column of y_true and y_pred
        accuracy = accuracy_score(y_true[:,idx], y_pred[:,idx])

        # Update accuracy_results with accuracy
        accuracy_results.append(accuracy)

        # Take the mean of accuracy_results
        avg_accuracy = np.mean(accuracy_results)

        return avg_accuracy



def build_model():
    
    #Build the Pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier(n_estimators=1000, random_state=0)))
])
    
    
    
    def avg_accuracy_score(y_true, y_pred):
        """
        Assumes that the numpy arrays `y_true` and `y_pred` ararys 
        are of the same shape and returns the average of the 
        accuracy score computed columnwise. 

        y_true - Numpy array - An (m x n) matrix 
        y_pred - Numpy array - An (m x n) matrix 

        avg_accuracy - Numpy float64 object - Average of accuracy score
        """

        # initialise an empty list
        accuracy_results = []

        # for each column index in either y_true or y_pred
        for idx in range(y_true.shape[-1]):
            # Get the accuracy score of the idx-th column of y_true and y_pred
            accuracy = accuracy_score(y_true[:,idx], y_pred[:,idx])

            # Update accuracy_results with accuracy
            accuracy_results.append(accuracy)

        # Take the mean of accuracy_results
        avg_accuracy = np.mean(accuracy_results)

        return avg_accuracy
    
    parameters = [
    {
        #'clf__estimator__max_leaf_nodes': [50, 100, 200],
        #'clf__estimator__min_samples_split': [2, 3, 4],
    }
]

    cv = GridSearchCV(pipeline, param_grid=parameters,
                 scoring=avg_accuracy_score, 
                verbose=10, 
                return_train_score=True 
                 )
    return cv
    pass


def evaluate_model(model, X_test, Y_test, category_names):
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