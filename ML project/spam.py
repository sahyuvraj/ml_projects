import pandas as pd 
import numpy as np
df = pd.read_csv('new_spam.csv',delimiter=',')
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
#print(df.head)
x = df['Message']
y = df['spam']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
#print(X_train.shape) print(X_test.shape)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline



v = CountVectorizer()

X_train_cv = v.fit_transform(X_train.values) # change text into vector form of number
X_test_cv = v.transform(X_test)


model = MultinomialNB()
model.fit(X_train_cv, y_train)



y_pred = model.predict(X_test_cv)

#print(classification_report(y_test, y_pred))
'''
emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]

emails_count = v.transform(emails)  # change email text into number of counter of text in vector form
model.predict(emails_count)'''




'''


clf = Pipeline([
     ('vectorizer', CountVectorizer()),
     ('nb', MultinomialNB())   
])
clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)
#print("Accuracy:",np.mean((y_predict==y_test)))
#print(classification_report(y_test,y_predict))
'''

import pickle

pickle.dump(model,open("spam.pkl",'wb'))
pickle.dump(v,open("spam_count_vector.pkl",'wb'))


