#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import gensim.downloader as api
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Dropout
from keras.utils import to_categorical
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[2]:


# Load the dataset
data = pd.read_excel('Dataset.xlsx')
df = pd.DataFrame(data)
df['Text'] = df['Text'].apply(lambda x: x.lower() if isinstance(x, str) else x)
df


# ## Text pre-processing ##

# In[3]:


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


# In[4]:


# Tokenize the text data
tokenized_words = []
for row in df['Text']:
    tokenized_words.append(preprocess_text(row))


# In[5]:


# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to NOUN if no match is found


# In[6]:


lemmatized_sentences = []

for list in tokenized_words:
    pos_tags = pos_tag(list)
    lemmatized_sentence = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tags]
    lemmatized_sentences.append(lemmatized_sentence)


# In[7]:


sentences = [' '.join(words) for words in lemmatized_sentences]

# Create the dataframe
df_2 = pd.DataFrame(sentences, columns=['Text'])
df_2['Target'] = df['Target']
df_2


# In[8]:


# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

def sentiment_features(text):
    # Calculate TextBlob sentiment scores
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity
    
    # Calculate VADER sentiment scores
    vader_sentiment = sia.polarity_scores(text)
    vader_compound = vader_sentiment['compound']
    vader_positive = vader_sentiment['pos']
    vader_neutral = vader_sentiment['neu']
    vader_negative = vader_sentiment['neg']
    
    return [sentiment_polarity, sentiment_subjectivity, vader_compound, vader_positive, vader_neutral, vader_negative]


# In[9]:


first_set_of_features = np.array([sentiment_features(sentence) for sentence in sentences])


# In[10]:


model = api.load('fasttext-wiki-news-subwords-300')


# In[11]:


def get_word_vectors(words, model):
    vectors = {}
    for word in words:
        if word in model:  # Directly check in model
            vectors[word] = model[word]  # Directly access the vector
        else:
            vectors[word] = None  # Or handle missing words as you prefer
    return vectors

vector_list = []
for sentence in df_2['Text']:
    vector_list.append(get_word_vectors(sentence, model))


# In[12]:


average_vectors = []

# Iterate over each sentence's dictionary of word vectors in vector_list
for i, word_vectors in enumerate(vector_list):
    valid_vectors = [v for v in word_vectors.values() if v is not None]  # Filter out None values
    if valid_vectors:
        # Calculate the average vector for this sentence
        sentence_avg_vector = np.mean(valid_vectors, axis=0)
        average_vectors.append(sentence_avg_vector)
    else:
        print(f"Sentence {i} has no valid vectors and is skipped.")

# After the loop, `average_vectors` should hold one average vector per sentence
print(f"Total number of vectors: {len(average_vectors)}")

# Check the shape of the first vector to confirm it's 300-dimensional
if average_vectors:
    print(f"Shape of average_vectors[0]: {average_vectors[0].shape}")


# In[13]:


sentiment_feature_columns = [f'Sentiment_Feature_{i+1}' for i in range(first_set_of_features.shape[1])]
final_df = pd.DataFrame(average_vectors, columns=[f'Feature_{i}' for i in range(len(average_vectors[0]))])
for i, column_name in enumerate(sentiment_feature_columns):
    final_df[column_name] = first_set_of_features[:, i]
final_df['Target'] = df['Target']
final_df.head()


# In[14]:


x = final_df.drop(columns = 'Target')
y = final_df['Target']


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=42)


# In[16]:


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# Normalize the test data using the same scaler
x_test_scaled = scaler.transform(x_test)


# ## KNN Model ##

# In[17]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score


# In[18]:


knn_classifier = KNeighborsClassifier(n_neighbors = 5)

# Train the model
knn_classifier.fit(x_train_scaled, y_train)

# Make predictions
y_pred_2 = knn_classifier.predict(x_test_scaled)


# In[19]:


print("Accuracy:", accuracy_score(y_test, y_pred_2))
print("Classification Report:\n", classification_report(y_test, y_pred_2))


# ## Support Vector Machine Model ##

# In[20]:


from sklearn.svm import SVC
from joblib import dump


# In[21]:


svm_classifier = SVC(kernel = 'linear', C = 0.3, random_state = 42)

# Train the model
svm_classifier.fit(x_train_scaled, y_train)

# Make predictions
y_pred = svm_classifier.predict(x_test_scaled)


# In[22]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[23]:


#dump(svm_classifier, 'svm_classifier.joblib')


# In[ ]:




