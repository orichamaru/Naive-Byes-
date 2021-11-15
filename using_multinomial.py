#ML Assign Naive Bayes Classifier 6
#Author - IIT2018501
#Multinomial Distribution

import numpy as np
import pandas as pd
import math

df = pd.read_csv('SMSSpamCollection', sep='\t',header=None,  names=['Type', 'Msg'])
x = df['Msg']
y = df['Type']

train_x = x[:3900]
train_y = y[:3900]

#Split words of Training message
train_x = train_x.str.replace( '\W', ' ')
train_x = train_x.str.lower()
train_x = train_x.str.split()

test_x = x[3900: ]
test_y = y[3900: ]

#Split words of Testing message
test_x = test_x.str.replace( '\W', ' ')
test_x = test_x.str.lower()
test_x = test_x.str.split()

#Creating dictionary from given set of messages
dictionary = []
for message in train_x:
   for word in message:
      dictionary.append(word)

#Converting list to set and then back to list containing unique words of messages
dictionary = list(set(dictionary))

#Dictionary containing count of each word of dictionary in given message
count_word = {word: [0]*len(train_x) for word in dictionary}

#Marking frequency of particular word in particular message
for index, message in enumerate(train_x):
   for word in message:
      count_word[word][index] += 1
word_count = pd.DataFrame(count_word)
word_count.head()

#xf contains final merged table
xf = pd.concat([df[:3900], word_count], axis=1)

#pspam - contains probability of spam messages
#pham - contains probability of ham messages
pspam = xf['Type'].value_counts()['spam']/xf.shape[0]
pham = xf['Type'].value_counts()['ham']/xf.shape[0]

#Number of words in spam message
nspam = xf.loc[xf['Type'] == 'spam','Msg'].apply(len).sum()
#Number of words in ham message
nham = xf.loc[xf['Type'] == 'ham','Msg'].apply(len).sum()
#Size of dictionary
ndict = len(dictionary)
# Laplace Smoothing
alpha = 1

# Initialise spam , ham list
spam_list = {word:0 for word in dictionary}
ham_list = {word:0 for word in dictionary}

# Calculate values of list
for word in dictionary:
   spam_list[word] = (xf.loc[xf['Type'] == 'spam', word].sum() + alpha) / (nspam + alpha*ndict)
   ham_list[word] = (xf.loc[xf['Type'] == 'ham', word].sum() + alpha) / (nham + alpha*ndict)

#Classify Function
def classify(message):
    pspam_message = pspam
    pham_message = pham
    for word in message:
        if word in spam_list:
          pspam_message *= spam_list[word]
        if word in ham_list: 
          pham_message *= ham_list[word]
    if pham_message >= pspam_message:
        return 'ham'
    elif pham_message < pspam_message:
        return 'spam'

#Testing on testing data
count=0
for message , mtype in zip(test_x, test_y):
   if(classify(message) == 'spam' and mtype =='spam'):
     count+=1
   elif(classify(message) == 'ham' and mtype =='ham'):
    count+=1
total = len(test_y)
print("Accuracy using Multinomial Distribution is " + str((count/total)*100))