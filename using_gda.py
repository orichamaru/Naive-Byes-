#ML Assign Naive Bayes Classifier 6
#Author - IIT2018501
#Gaussian Disciminant Analysis 

import numpy as np
import pandas as pd
import math

def find_px_py(x,mu,sigma):
   dn = pow(2*np.pi,0.5)*pow(sigma,1)
   nmp = ((x-mu)*(x-mu))/(sigma*sigma)
   return np.exp(-0.5*nmp)/dn

df = pd.read_csv('SMSSpamCollection', sep='\t',header=None,  names=['Type', 'Msg'])
x = df['Msg']
y = df['Type']

train_x = x[:3900]
train_y = y[:3900]

#size 
m = 3900

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

#xf contains final merged table
xf = pd.concat([df[:3900], word_count], axis=1)

#pspam - contains probability of spam messages
#pham - contains probability of ham messages
pspam = xf['Type'].value_counts()['spam']/xf.shape[0]
pham = xf['Type'].value_counts()['ham']/xf.shape[0]

#Number of words in spam message
no_spam = xf.loc[xf['Type'] == 'spam','Msg'].apply(len).sum()
#Number of words in ham message
no_ham = xf.loc[xf['Type'] == 'ham','Msg'].apply(len).sum()
#Size of dictionary
ndict = len(dictionary)
# Laplace Smoothing
alpha = 1

# Initialise spam , ham list
spam_list = {word:{} for word in dictionary}
ham_list = {word:{} for word in dictionary}


# Calculate mu, sigma of each word in dictionary
for word in dictionary:
   sigma0, sigma1 = 0,0
   mu1 = (xf.loc[xf['Type'] == 'spam', word].sum()) / no_spam
   for j in xf.loc[xf['Type'] == 'spam', word]:
     sigma1+=(j-mu1)**2
   mu0 = (xf.loc[xf['Type'] == 'ham', word].sum()) / no_ham
   for j in xf.loc[xf['Type'] == 'ham', word]:
     sigma0+=(j-mu0)**2
   spam_list[word]['mu'] = mu1
   spam_list[word]['sigma'] = sigma1/m
   ham_list[word]['mu'] = mu0
   ham_list[word]['sigma'] = sigma0/m

#Classify Function
def classify(message):
    pspam_message = pspam
    pham_message = pham
    for word in message:
        freq = dictionary.count(word)
        if word in spam_list and spam_list[word]['sigma']:
          pspam_message *= find_px_py(freq, spam_list[word]['mu'], spam_list[word]['sigma'])
        if word in ham_list and ham_list[word]['sigma']: 
          pham_message *= find_px_py(freq, ham_list[word]['mu'], ham_list[word]['sigma'])
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
print("Accuracy using Gaussian is " + str((count/total)*100))