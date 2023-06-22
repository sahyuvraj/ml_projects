import pandas as pd 
car = pd.read_csv("car.csv",delimiter=",")
# we have to clean the data because our dataset having more irregularity
# in data set year,kms and price in object and not clean  we need to clean and  convert into int and data having also having null value
'''in dataset year has many non year value,year object to int 
  price object to int 
  we have to remove kms into kms value and convert object into int 
  fuel values has nan values we have to remove that null values
  name attributes is to long we have minimize them
  you can see the error through:- print(car['kms_drive'].unique()) 
  same you can see for all the column
'''
# first do data cleaning 
backup = car.copy()

car = car[car['year'].str.isnumeric()]
#print(car.info) #print(car['year'].unique)
car['year'] = car['year'].astype(int)
#print(car.info) #print(car['year']) #print(car['price'])
car = car[car['Price']!="Ask For Price"]
#in price attribute comma is there we have to remove the commma and convert into integer
#print(car['Price'])
car['Price'] = car['Price'].str.replace(',','').astype(int)
#print(car['Price'])
car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',','')
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)
#print(car['kms_driven'])
car = car[~car['fuel_type'].isna()]
car['name'] = car['name'].str.split(' ').str.slice(0,3).str.join(' ')
#print(car.head)
car = car.reset_index(drop=True)
#print(car.info) #print(car.describe())
car = car[car['Price']<6e6].reset_index(drop=True)
#print(car.shape)
#print(car.describe())
car.to_csv('cleaned car.csv')

x = car.drop(columns='Price')
y = car['Price']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

ohe = OneHotEncoder()
ohe.fit(x[['name','company','fuel_type']])
column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')
lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)


y_predict=pipe.predict(x_test)
#print(y_predict) #print(y_predict[0][0]) 
#print(r2_score(y_test,y_predict))
import pickle
pickle.dump(pipe,open('lrmodel.pkl','wb'))
pickle.dump(car,open('df.pkl','wb'))




