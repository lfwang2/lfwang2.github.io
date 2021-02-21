import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

data = pd.read_csv(r'boston_housing_data.csv')
data = data.replace('?',np.nan)
data = data.dropna(how='any')
class LinearRegression:
    def __init__(self, alpha,times):
        self.alpha = alpha
        self.times = times
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.w_ = np.zeros(1 + X.shape[1])
        #print('self.w_:\n',self.w_)
        self.loss = []
        for i in range(self.times):
            y_hat = np.dot(X, self.w_[1:]) +self.w_[0]
            error = y - y_hat
            self.loss.append(np.sum(error**2)/2)
            self.w_[0] += self.alpha*np.sum(error)
            self.w_[1:] += self.alpha * np.dot(X.T, error)
            #print(i)
    def predict(self, X):
        X = np.asarray(X)
        result = np.dot(X,self.w_[1:] + self.w_[0])
        return result
class StandardScaler:
    def fit(self,X):
        X = np.asarray(X)
        self.std_ = np.std(X, axis=0)
        self.mean_ =np.mean(X, axis=0)

    def transform(self,X):
        return (X -self.mean_)/self.std_
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
lr = LinearRegression(alpha=0.0005, times=100)
t = data.sample(len(data),random_state=0)
train_X = t.iloc[:400,:-1]
train_y = t.iloc[:400,-1]
test_X = t.iloc[400:,:-1]
test_y = t.iloc[400:,-1]

s = StandardScaler()
train_X =s.fit_transform(train_X)
test_X = s.transform(test_X)

s2 = StandardScaler()
train_y = s2.fit_transform(train_y)
test_y = s2.transform(test_y)

lr.fit(train_X, train_y)
result = lr.predict(test_X)
print(np.mean((result-test_y)**2))
print('lr.w_:\n',lr.w_)
print('lr.loss:\n',lr.loss)

plt.figure(figsize=(24,8))
plt.plot(result, 'ro-',label='PD')
plt.plot(test_y.values,'go--',label='TD')
plt.title('LinearRegression-GD')
plt.xlabel('SN')
plt.ylabel('HP')
plt.legend()
plt.plot(range(1,lr.times+1),lr.loss,'o-')
plt.show()