import pandas as pd


dataset = pd.read_csv('knn.csv')
#print(dataset.shape) print(dataset.head)
#print(dataset.dtypes)

x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values



#print(x.shape) print(y.shape)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
k =5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)



'''
from collections import Counter

def predict_one(x_train,y_train,x_test,k):
    distances = []

    for i in range (len(x_train)):
        distance = ((x_train[i,:] - x_test)**2).sum()
        distances.append([distance,i])
    
    distances = sorted(distances)

    targets = []
    for i in range(k):
        training_index = distances[i][1]
        targets.append(y_train[training_index])
    
    return Counter(targets).most_common(1)[0][0]
        



def predict(x_train,y_train,x_test_data,k):
    prediction=[]

    for x_test in x_test_data:
        prediction.append(predict_one(x_train,y_train,x_test,k))
    return prediction

y_predict = predict(x_train,y_train,x_test,7)
'''





import pickle
data = {"model": clf}
pickle.dump(data,open('knn1.pkl','wb'))


