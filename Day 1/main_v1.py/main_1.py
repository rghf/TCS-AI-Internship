#We import the basic libraries and then read the dataset
import pandas as pd
import numpy as np
data = pd.read_csv('text_emotion.csv')
#simple dataset has four columns, the tweet ID, emotion depicted by the tweet, the author, and the text content of the tweet.
# We do not necessarily need the author column. Hence we can drop it
data = data.drop('author', axis=1)
# Dropping rows with other emotion labels
data = data.drop(data[data.sentiment == 'anger'].index)
data = data.drop(data[data.sentiment == 'boredom'].index)
data = data.drop(data[data.sentiment == 'enthusiasm'].index)
data = data.drop(data[data.sentiment == 'empty'].index)
data = data.drop(data[data.sentiment == 'fun'].index)
data = data.drop(data[data.sentiment == 'relief'].index)
data = data.drop(data[data.sentiment == 'surprise'].index)
data = data.drop(data[data.sentiment == 'love'].index)
data = data.drop(data[data.sentiment == 'hate'].index)
data = data.drop(data[data.sentiment == 'neutral'].index)
data = data.drop(data[data.sentiment == 'worry'].index)