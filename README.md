### SOCIAL MEDIA SENTIMENT ANALYSIS & CLASSIFICATION

# 1. Project Overview & Objective
Interviewer-Style Explanation: "My project is a comprehensive Social Media Sentiment Analysis and Classification system. The core objective was twofold: first, to perform an in-depth Exploratory Data Analysis (EDA) 
on social media posts to understand user sentiment trends across various platforms, countries, and times; and second, to build and evaluate multiple Machine Learning classification models to accurately predict 
the sentiment (Positive, Negative, Neutral) of new, unseen text data."

=>The complete project workflow is contained within the main.ipynb Jupyter Notebook

# 2. Technical Stack & Libraries
This project was developed entirely in Python and utilizes a standard, robust set of libraries for Data Science and NLP:
**Data Manipulation:** Pandas, NumPy
**Text Processing:** NLTK and re for Tokenization, Stopword Removal, Stemming (PorterStemmer), and Regex for text cleaning.
**Visualization:** Matplotlib, Seaborn, WordCloud for Creating clear visualizations for EDA (sentiment distribution, temporal trends).
**Machine Learning:** Scikit-learn (sklearn) for Model selection, training, evaluation, cross-validation, and hyperparameter tuning (GridSearchCV).

# 3. Methodology: The 4-Stage NLP Pipeline
I followed a structured, four-stage pipeline, which is a common best practice in NLP projects.

## A. Data Preprocessing & Cleaning
**Initial Cleaning:** Removed noise from the text data (e.g., URLs, HTML tags, special characters) and standardized the text to lowercase.

**Text Transformation:** Applied core Natural Language Processing (NLP) techniques:

**Tokenization:** Breaking text into individual words.

**Stopword Removal:** Eliminating common, uninformative words (e.g., 'the', 'a', 'is').

**Stemming:** Reducing words to their root form (e.g., 'running' to 'run') using the Porter Stemmer.

**Feature Vectorization:** Converted the cleaned text into a numerical format using TF-IDF Vectorization (Term Frequency-Inverse Document Frequency) to be processed by ML models.

# B. Exploratory Data Analysis (EDA)
**Sentiment Distribution:** Visualized the overall counts of Positive, Negative, and Neutral posts.

**Temporal Analysis:** Analyzed post frequency to determine the Peak Posting Hour and sentiment trends over time.

**Platform & Country Analysis:** Determined which platforms and countries contribute most to each sentiment type.

**Engagement Analysis:** Calculated the correlation between Likes and Retweets.

# C. Model Training & Selection
**Models Evaluated:** The performance of three classic, yet powerful, classification algorithms was compared:

1. Logistic Regression

2. Random Forest Classifier

3. Support Vector Classifier (SVC)

**Evaluation:** Employed Cross-Validation and Grid Search to ensure robust performance, prevent overfitting, and optimize the best model's hyperparameters.

# 4. Key Results & Insights
The final summary provided several key business and technical insights, which are critical for any stakeholder interested in social media strategy:

**Best Model:** Highest-performing classification algorithm based on accuracy and F1-Score.
**Model Accuracy:** Quantified performance of the best model (e.g., ~90% Accuracy).
**Dominant Sentiment:** The most frequent overall sentiment in the dataset.
**Peak Hour:** The hour of the day when posting activity is highest.
**Best Platform:** The platform with the highest engagement or most positive sentiment.
**Likes/Retweets Correlation:** The linear relationship between likes and retweets, indicating engagement consistency.
