Text Classification and Sentiment Analysis
Overview
This project involves processing text data for sentiment analysis and classification. The script performs several steps including text preprocessing, sentiment feature extraction, word vector averaging, and model training using machine learning algorithms. The final goal is to classify text into predefined categories based on sentiment and other features.

Dependencies
The script requires the following Python libraries:

pandas
numpy
nltk
gensim
keras
textblob
scikit-learn
joblib
You can install the required libraries using pip:
pip install pandas numpy nltk gensim keras textblob scikit-learn joblib

Data
The dataset used is Dataset.xlsx, which should be placed in the same directory as the script. The dataset must contain at least two columns:

Text: The text data to be analyzed.
Target: The target labels for classification.

**Script Overview**

Load and Preprocess Data:

Load the dataset from an Excel file.
Convert text to lowercase and preprocess it by tokenizing and lemmatizing.

Feature Extraction:

Use NLTK's VADER and TextBlob for sentiment analysis.
Generate word vectors using the FastText model and compute average vectors for each sentence.
Model Training and Evaluation:

Split the data into training and testing sets.
Normalize the data.
Train a K-Nearest Neighbors (KNN) classifier and a Support Vector Machine (SVM) classifier.
Evaluate model performance using accuracy and classification reports.
How to Run the Script

Prepare the Dataset:
Ensure Dataset.xlsx is in the same directory as the script.
Run the Script:

Execute the script using Python
Results:

The script prints accuracy and classification reports for both KNN and SVM classifiers.
Sample Output
After running the script, you should see output similar to the following:

plaintext
Copy code
Accuracy: 0.85
Classification Report:
              precision    recall  f1-score   support

       class1       0.84      0.88      0.86       150
       class2       0.86      0.82      0.84       150

    accuracy                           0.85       300
   macro avg       0.85      0.85      0.85       300
weighted avg       0.85      0.85      0.85       300
Notes
Ensure you have the necessary NLTK data downloaded. You can install them using:


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
The script uses gensim for word vectors, which requires downloading the FastText model. Ensure you have a stable internet connection.

Contact
If you have any questions or feedback, feel free to contact me at s.kamvisis1@gmail.com
