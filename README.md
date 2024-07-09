# Disaster Tweet Classification

This project is part of a Kaggle competition aimed at predicting whether a given tweet is about a disaster or a normal event. The solution involves preprocessing the tweet text data and training a Long Short-Term Memory (LSTM) model to classify tweets.

## Overview

The goal of this project is to build a machine learning model that can accurately classify tweets as either mentioning a disaster or not. This involves preprocessing the text data, tokenizing it, and using an LSTM neural network for classification.

## Dataset

The dataset used in this project is provided by Kaggle and consists of the following files:
- `train.csv`: The training set containing tweets and their corresponding labels.
- `test.csv`: The test set used for making predictions (labels are not provided).

The `train.csv` file has the following columns:
- `id`: Unique identifier for each tweet
- `text`: The tweet text
- `target`: The label indicating whether the tweet is about a disaster (1) or not (0)

## Installation

To run this project, you need to have Python installed along with the following libraries:
- pandas
- numpy
- nltk
- tensorflow
- matplotlib
- seaborn


Additionally, you need to download the NLTK stopwords and wordnet datasets:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## Preprocessing

1. **Lowercasing**: Convert all characters in the tweets to lowercase.
2. **Removing Punctuation and Digits**: Remove all punctuation and digits from the tweets.
3. **Tokenization**: Split the tweets into individual words (tokens).
4. **Stopwords Removal**: Remove common stopwords that do not contribute to the meaning of the tweets.
5. **Lemmatization**: Convert words to their base form using WordNet lemmatizer.

## Model

We use an LSTM model for classification. The architecture of the model is as follows:
- An Embedding layer to convert words to vectors.
- An LSTM layer with dropout and recurrent dropout for regularization.
- A Global Max Pooling layer to reduce the dimensions.
- A Dense layer with ReLU activation and L2 regularization.
- A Dropout layer for further regularization.
- A Dense output layer with sigmoid activation for binary classification.

## Training

The model is trained using the following parameters:
- Optimizer: Adam with a learning rate of 0.001
- Loss function: Binary Crossentropy
- Metrics: Accuracy
- Number of epochs: 30

The data is split into training and validation sets to monitor the model's performance during training.


This project is part of a Kaggle competition. We thank Kaggle for providing the dataset and the platform for this competition.

For more details, visit the [Kaggle competition page](https://www.kaggle.com/c/nlp-getting-started).
