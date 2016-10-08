from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors

clf = tree.DecisionTreeClassifier()

## CHALLENGE - create 3 more classifiers...
#1
clf1 = svm.SVC()
#2
clf2 = GaussianNB()
#3
clf3 = neighbors.KNeighborsClassifier()

#[height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']


#CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)

#Support Vector classifier
clf1 = clf1.fit(X,Y)

#Naive Bayes
clf2 = clf2.fit(X,Y)

#K neighbors classifier
clf3 = clf3.fit(X,Y)
#test data

X_test=[[198,92,48],[184,84,44],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
Y_test=['male','male','male','female','female','female','male','male']

Y_prediction1 = clf1.predict(X_test)

Y_prediction2 = clf2.predict(X_test)

Y_prediction3 = clf3.predict(X_test)

#CHALLENGE compare their reusults and print the best one!

print("Prediction for Naive Bayes : ",Y_prediction2)

print("Accuracy for Naive Bayes : ",accuracy_score(Y_test,Y_prediction2))

print("Accuracy for SVM : ",accuracy_score(Y_test,Y_prediction1))

print("Accuracy for K neighbors : ",accuracy_score(Y_test,Y_prediction3))



