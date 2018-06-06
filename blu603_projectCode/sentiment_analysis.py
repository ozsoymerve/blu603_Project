from __future__ import print_function, division
from future.utils import iteritems
from builtins import range


import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup


wordnet_lemmatizer = WordNetLemmatizer()

stopwords = set(w.rstrip() for w in open('stopwords.txt')) #not included

positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text') #take just <review_text> parts of all dataset

negative_reviews = BeautifulSoup(open('electronics/negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')

diff = len(positive_reviews) - len(negative_reviews) # there are more positive reviews than negative reviews
#print(diff) #just to look at
rand = np.random.choice(len(negative_reviews), size=diff) #we will add some values for negative_reviews
extra = [negative_reviews[i] for i in rand]
#print(negative_reviews)
negative_reviews += extra
#print(negative_reviews)


def my_tokenizer(s): #to seperating sentences with words
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2] # remove short words , in generally they have not any sentiment
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords] # we should get stopwords out of tokens
    return tokens #we have a useful tokens


# create a word-to-index map so that we can create our word-frequency vectors later
# let's also save the tokenized versions so we don't have to tokenize again later
word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []

for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1


# now let's create our input matrices
def tokens_to_vector(tokens, label): ##take each tokens and create a data array which is going to be a bunch of number
  # we want to shuffle train and test sets , put both labels and vector into the same array, makes easier to shuffle
    x = np.zeros(len(word_index_map) + 1) # add 1  for the label
    for t in tokens: #take first tokesn and keep going
        i = word_index_map[t] #I get the index for the index map
        x[i] += 1
    x = x / x.sum() # normalize it before setting label
  #x arrayında tüm indexlerin normalizasyonu var
    x[-1] = label #last element
    return x


N = len(positive_tokenized) + len(negative_tokenized) #total number
data = np.zeros((N, len(word_index_map) + 1)) #initilaze the array
i = 0 #this counter so, I can keep track of which sample I am looking at
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1) #lebels=1 because positive reviews
#    print("positive")
#    print(xy)
    data[i,:] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
#    print("negative")
#    print(xy)
    data[i,:] = xy
    i += 1

# shuffle the data and create train/test splits
np.random.shuffle(data) #np.random.shuffle: The array or list to be shuffled.

X = data[:,:-1] #all rows  except the last column
Y = data[:,-1] #for labels

# last 100 rows will be test
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

#create the model with logistic Regression
model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print("Classification rate:", model.score(Xtest, Ytest))
#our classificationn rate on the test set

# weights for each word
threshold = 0.5 #exchangeable
for word, index in iteritems(word_index_map):
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
     print(word, weight)  ##so you see that some of these make sense

