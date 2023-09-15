import pickle
from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
X, y = datasets.load_iris(return_X_y=True)

clf.fit(X, y)

with open('./pickled_model.pickle', 'wb') as file:
    pickle.dump(clf, file)
