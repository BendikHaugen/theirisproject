import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Check the versions of libraries
def check_version():
    print('Python: {}'.format(sys.version)) #python verison
    print('scipy: {}'.format(scipy.__version__)) #Scipy version
    print('numpy: {}'.format(numpy.__version__)) #numpy verson
    print('matplotlib: {}'.format(matplotlib.__version__)) #matplotlib version
    print('pandas: {}'.format(pandas.__version__))#pandas version
    print('sklearn: {}'.format(sklearn.__version__))#sklean version

#check_version()
#loading the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#uncomment to view dataset
#print(dataset)
#dataset := [150rows x 5columns]
#print(dataset.describe())
#print(dataset.groupby('class').size())

def box_plot():
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    matplotlib.pyplot.show()

#box_plot()
#view both the historgram and the scatterplot
#scatter_matrix(dataset)
#pyplot.show()

'''The idea is to hold back som data from the algorithms
and we will use this data to get an independent idea of how 
accurate the best model actually is
The loaded dataset are split into two, 80% is used
to train, and 20% will be held back as a validation
dataset'''

array = dataset.values
x = array[:, 0:4]
y = array[:, 4]
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=1)

'''
models
Logistic regression = LR
Linear descriminant analysis = LDA
K-neares neighbors = KNN
Classification and regression trees = CART
Gaussian Naive Bayes = NB
Suppert Vector = SCM
'''
Models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
('LDA', LinearDiscriminantAnalysis()),
('KNN', KNeighborsClassifier()),
('CART', DecisionTreeClassifier()),
('NB', GaussianNB()),
('SVM', SVC(gamma='auto'))]

#evaluating the models

results, names = [], []
for name, model in Models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#comparing algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

#making predictions
model = SVC(gamma='auto')
model.fit(x_train, y_train)
predictions = model.predict(x_validation)

#evaluate predictions
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))
