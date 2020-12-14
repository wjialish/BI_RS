import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

path = 'mnist.npz'
#f = np.load(path)
f = np.load(path)

X_train, y_train = f['x_train'], f['y_train']
X_test, y_test = f['x_test'], f['y_test']

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255.
X_test /= 255.

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

roc_Decision = 0
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

sum = 0.0
for i in range(10000):
    if (y_pred[i] == y_test[i]):
        sum = sum + 1

print('Test set score: %f' % (sum / 10000.))

# Test set score: 0.877100