import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

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
print(dataset)