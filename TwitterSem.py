import re
import pickle
import pandas as pd
# plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# nltk
from nltk.stem import WordNetLemmatizer

# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report



df = pd.read_csv('/home/nawaf/Downloads/Twitter/training.1600000.processed.noemoticon.csv',encoding='ISO-8859-1',names=["sentiment", "ids", "date", "flag", "user", "text"])
print(df.head())
df=df[['sentiment','text']]
df['sentiment']=df["sentiment"].replace(4,1)
print(df.head())

ax = df.groupby('sentiment').count().plot(kind='bar', title='Distribution of data',
                                               legend=False)
ax.set_xticklabels(['Negative','Positive'], rotation=0)

# Storing data in lists.
text, sentiment = list(df['text']), list(df['sentiment'])