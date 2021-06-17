# Disaster Response Pipeline Project



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


The goal of this project is to build site that can classify disaster messages and help emergancy forces to better alocate resources. The project uses figure 8 data to train a model that help identify messages as soon once the user enters it.



### Built With

* [Python](https://www.python.org/downloads/)
* [sys](https://docs.python.org/3/library/sys.html)
* [pandas](https://pandas.pydata.org/)
* [numpy](https://numpy.org/)
* [NLTK](https://www.nltk.org/)
* [sklearn](https://sklearn.org/)
* [html.parser](https://docs.python.org/3/library/html.parser.html)
* [plotly](https://plotly.com/python/)
* [flask](https://flask.palletsprojects.com/en/2.0.x/)
* [sqlalchemy](https://www.sqlalchemy.org/)




<!-- GETTING STARTED -->
## Getting Started




### Prerequisites


* Python 3.5+
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Model Loading and Saving Library: Pickle, joblib
* Web App and Data Visualization: Flask, Plotly

Installation

### Installation

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
   ```


<!-- LICENSE -->
## License

The files are free to use 


<!-- CONTACT -->
## Contact

Project Link: [https://github.com/Hangzoed/Disaster-Response-Pipeline](https://github.com/Hangzoed/Disaster-Response-Pipeline)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [Thanks for Firgure 8 for the Dataset](https://www.figure-eight.com/)
* [Thanks for Udacity Data Science program for providing initial file structure page layout and amazing mentors](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
* [I learned alot from  sarasun97](https://github.com/sarasun97)
* [I learned alot from alirezakfz ](https://github.com/alirezakfz)
* [I learned a lot from evansdoe/](https://github.com/evansdoe/disaster-response-pipeline)
* [README template from othneildrewl It helped me to accelrate the README](https://github.com/othneildrew/Best-README-Template)





