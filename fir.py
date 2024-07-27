
### 1. Importing the Libraries
import pandas as pd

data = pd.read_csv('heart.csv')

# 3. Taking Care of Missing Values
data.isnull().sum()

### 4. Taking Care of Duplicate Values
data_dup = data.duplicated().any()
data_dup ##this give the o/p TRUE bcoz data contain duplicate
data = data.drop_duplicates() ##drop the duplicate value
data_dup = data.duplicated().any()
data_dup


### 5. Data Processing
cate_val = [] #list1 for storing value of the catogorical coloumn
cont_val = [] #list1 for storing value of the numerical coloumn
for column in data.columns:
    if data[column].nunique() <=10:
        cate_val.append(column)
    else:
        cont_val.append(column)
# print(cate_val)
# print(cont_val)


### 6. Encoding Categorical Data
cate_val
data['cp'].unique()
cate_val.remove('sex')
cate_val.remove('target')
data = pd.get_dummies(data,columns = cate_val,drop_first=True)
data.head()

### 7. Feature Scaleing allow us to put the featured in the same scale 
data.head()#before feature scaling 
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
data[cont_val] = st.fit_transform(data[cont_val])
data.head()#after features scaling


# 8. Splitting The Dataset Into The Training Set And Test Set
X = data.drop('target',axis=1)
y = data['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,
                                              random_state=42)
# X_test
# y_train
# X_train
# y_test



### 9. Logistic Regression
data.head()
# our target is catogorical so we used LR

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)#trainng

y_pred1 = log.predict(X_test)#testing
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred1) ##checking the accuracy 

# ### 10. SVC
# from sklearn import svm
# svm = svm.SVC()
# svm.fit(X_train,y_train)
# y_pred2 = svm.predict(X_test)
# accuracy_score(y_test,y_pred2)


### 11. KNeighbors Classifier

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()#k=5
knn.fit(X_train,y_train)
y_pred3=knn.predict(X_test)
accuracy_score(y_test,y_pred3)



score = []
#checking on which vzlue of k it work more accurate
for k in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    score.append(accuracy_score(y_test,y_pred))
# print(score) it will print the lsit of the accuracy





 #graph 
# import matplotlib.pyplot as plt
# plt.plot(score)
# plt.xlabel("K Value")
# plt.ylabel("Acc")
# plt.show()
# knn=KNeighborsClassifier(n_neighbors=2)
# knn.fit(X_train,y_train)
# y_pred=knn.predict(X_test)
# accuracy_score(y_test,y_pred)



### Non-Linear ML Algorithms here wedont need the feature scaling and data scaling
data = pd.read_csv('heart.csv')
data = data.drop_duplicates()
X = data.drop('target',axis=1)
y=data['target']
#spliting the data
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,
                                                random_state=42)

### 12. Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred4= dt.predict(X_test)
accuracy_score(y_test,y_pred4)

### 13. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred5= rf.predict(X_test)
accuracy_score(y_test,y_pred5)


# final dat
final_data = pd.DataFrame({'Models':['LR','KNN','DT','RF'],
                          'ACC':[accuracy_score(y_test,y_pred1)*100,
                                
                                accuracy_score(y_test,y_pred3)*100,
                                accuracy_score(y_test,y_pred4)*100,
                                accuracy_score(y_test,y_pred5)*100,]})
print(final_data)


# prnting the barplot of the diferent 
# import seaborn as sns
# sns.barplot(final_data['Models'],final_data['ACC'])

X=data.drop('target',axis=1)
y=data['target']
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X,y)

### 15. Prediction on New Data

from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
log.fit(X,y)

### 15. Prediction on New Data

import pandas as pd
new_data = pd.DataFrame({
    'age':52,
    'sex':1,
    'cp':0,
    'trestbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalach':168,
    'exang':0,
    'oldpeak':1.0,
     'slope':2,
    'ca':2,
    'thal':3,    
},index=[0])
print(new_data)
p = log.predict(new_data)  # Use Logistic Regression for prediction
if p[0]==0:
    print("No Disease")
else:
    print("Disease")
    
### 16. Save Model Using Joblib
import joblib
joblib.dump(rf,'model_joblib_heart')
model = joblib.load('model_joblib_heart')
print(model.predict(new_data))
data.tail()

### GUI
from tkinter import *
import joblib
def show_entry_fields():
    p1=int(e1.get())
    p2=int(e2.get())
    p3=int(e3.get())
    p4=int(e4.get())
    p5=int(e5.get())
    p6=int(e6.get())
    p7=int(e7.get())
    p8=int(e8.get())
    p9=int(e9.get())
    p10=float(e10.get())
    p11=int(e11.get())
    p12=int(e12.get())
    p13=int(e13.get())
    model = joblib.load('model_joblib_heart')
    result=model.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]])
    
    if result == 0:
        Label(master, text="No Heart Disease").grid(row=31)
    else:
        Label(master, text="Possibility of Heart Disease").grid(row=31)
    
    
master = Tk()
master.title("Heart Disease Prediction System")


label = Label(master, text = "Heart Disease Prediction System"
                          , bg = "black", fg = "white"). \
                               grid(row=0,columnspan=2)


Label(master, text="Enter Your Age").grid(row=1)
Label(master, text="Male Or Female [1/0]").grid(row=2)
Label(master, text="Enter Value of CP").grid(row=3)
Label(master, text="Enter Value of trestbps").grid(row=4)
Label(master, text="Enter Value of chol").grid(row=5)
Label(master, text="Enter Value of fbs").grid(row=6)
Label(master, text="Enter Value of restecg").grid(row=7)
Label(master, text="Enter Value of thalach").grid(row=8)
Label(master, text="Enter Value of exang").grid(row=9)
Label(master, text="Enter Value of oldpeak").grid(row=10)
Label(master, text="Enter Value of slope").grid(row=11)
Label(master, text="Enter Value of ca").grid(row=12)
Label(master, text="Enter Value of thal").grid(row=13)



e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)
e12 = Entry(master)
e13 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)
e12.grid(row=12, column=1)
e13.grid(row=13, column=1)



Button(master, text='Predict', command=show_entry_fields).grid()

mainloop()

