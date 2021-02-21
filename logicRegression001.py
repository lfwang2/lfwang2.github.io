import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
data=pd.read_csv('iris.data',header=None)
data = data.replace('?', np.nan)
data = data.dropna(how='any')
data = data.drop_duplicates()
data[4] = data[4].map({'Iris-versicolor':0,'Iris-setosa':1,'Iris-virginica':2})
data = data[data[4] !=2]

class LogisticRegression:
    def __init__(self,alpha,times):
        self.alpha = alpha
        self.times = times
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
    def fit(self,X,y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.w_ =np.zeros(1+X.shape[1])
        self.loss_=[]
        for i in range(self.times):
            z = np.dot(X,self.w_[1:]) + self.w_[0]
            p = self.sigmoid(z)
            cost = -np.sum(y*np.log(p) +(1-y)*np.log(1-p))
            self.loss_.append(cost)
            self.w_ += self.alpha*np.sum(y-p)
            self.w_[1:] += self.alpha*np.dot(X.T,y-p)
    def predict_proba(self,X):
        X=np.asarray(X)
        z=np.dot(X,self.w_[1:])+self.w_[0]
        p=self.sigmoid(z)
        p=p.reshape(-1,1)
        print(p)
        return np.concatenate([1-p, p],axis=1)
    def predict(selfself,X):

         return np.argmax(self.predict_proba(X), axis=1)


t1 = data[data[4]==0]
t2 = data[data[4]==1]
t1 = t1.sample(len(t1),random_state=0)
t2 = t2.sample(len(t2),random_state=0)

train_X = pd.concat([t1.iloc[:40,:-1],t2.iloc[:40,:-1]],axis=0)
train_y = pd.concat([t1.iloc[:40,-1],t2.iloc[:40,-1]],axis=0)
test_X = pd.concat([t1.iloc[40:,:-1],t2.iloc[40:,:-1]],axis=0)
test_y = pd.concat([t1.iloc[40:,-1],t2.iloc[40:,-1]],axis=0)

lr = LogisticRegression(alpha=0.01,times=20)
lr.fit(train_X,train_y)
result = lr.predict_proba(test_X)
plt.plot(result,'ro',ms=10,label='PD')
plt.plot(test_y.values,'go',label='TD')
plt.title('LOGISTAICREGRESSION')
plt.xlabel('CASEID')
plt.ylabel('TYPE')
plt.legend()
plt.show()

