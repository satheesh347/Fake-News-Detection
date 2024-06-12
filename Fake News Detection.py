import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Change the path to the news.csv file if necessary
data = pd.read_csv("/content/sample_data/news.csv")

# Print the first few rows of the data
print(data.head())
print(data.head())


x = np.array(data["title"])
y = np.array(data["label"])

cv = CountVectorizer()
x = cv.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

news_headline = "Cow dung can cure Corona Virus"
data = cv.transform([news_headline]).toarray()
print(model.predict(data))

