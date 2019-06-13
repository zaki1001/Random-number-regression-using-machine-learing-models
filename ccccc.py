# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 18:06:09 2019

@author: zaki
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(10)
you=pd.DataFrame(np.random.randint(10,50,size=(25,20)),columns=list('ABCDEFGHIJKLMNOPQRST'))




f=[]
#jjj=you[0]
#jjj=list(map((lambda x : (x-10)/2.0 if x>25 else (x-5)),you['A']))

for i in range(0,len(you['A'])):
    for j in list('ABCDEFGHIJKLMNOPQRST'):
        if you.loc[i,j]>25:
            you.at[i,j]=(you.loc[i,j]-10)/2.0 
            
        elif you.loc[i,j]>20:
            you.at[i,j]=(you.loc[i,j]-5)/2.0 
            
        elif you.loc[i,j]>15:
            you.at[i,j]=(you.loc[i,j])/2.0 
            
        else:
            you.at[i,j]=abs(you.loc[i,j]-5)
            
            
#plt.scatter((you['A']),you['B'])

f=pd.DataFrame(np.random.randint(0,3,size=(20,1)))
f1=pd.DataFrame(np.random.randint(0,3,size=(5,1)),index=(20,21,22,23,24))
f=f.append(f1)

    
#from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC






X_train=you.iloc[:19,:4]
X_test=you.iloc[20:24,:4]
y_train=f.iloc[:19,:]
y_test=f.iloc[20:24,:]



#X=you.iloc[:,1:4]
#y=f.iloc[:,1:]

#
#X_train,X_test,y_train,y_test=t= train_test_split(X,y,test_size=0.25)
sv=SVC(kernel='linear')
sv.fit(X_train,y_train)
sv_pred=sv.predict(X_test)


lr=LogisticRegression()
lr.fit(X_train,y_train)
lr_pred=lr.predict(X_test)
print("accuraacy score is:",accuracy_score(y_test,lr_pred))
print("accuracy score for svc is :",accuracy_score(y_test,sv_pred))

plt.figure(figsize=(20,20))
plt.subplot(121)
plt.scatter(y_test,sv_pred,c=sv_pred)
plt.subplot(122)
plt.scatter(y_test,lr_pred,c=lr_pred)








        