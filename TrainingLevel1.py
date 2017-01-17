import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
from sklearn.metrics import roc_auc_score

########## Read CSV File ############
def readCSVFile(reading_directory,filename):
    data=pd.read_csv(reading_directory+'/'+filename)
    return data

########## Write Datframe Into CSVFile ############
def writeCSVFile(data,writing_directory,filename):
    data.to_csv(writing_directory+'/'+filename,index=False)   

########## Divide Dataframe Into Train and Test ############
def divideTrainTest(data):
    train=data.sample(frac=0.7,random_state=200)
    test=data.drop(train.index)
#   train, test = train_test_split(data, test_size = 0.2, stratify=data['Level1'])
    return (train,test)
    
######Converting Categorical To Numerical##############
def labelEncoding(data,var_mod):
    le = LabelEncoder()   
    for i in var_mod:
        data[i] = le.fit_transform(data[i])
    return data
    
######Building COnfsuion Matrix##############
def confusionMatrix(Y_actual,Y_test):
    return(confusion_matrix(Y_actual, Y_test))

##### Building Numpy Mtarix ##########
def numpyMatrix(data):
    numpyMatrix_data = data.as_matrix()
    x=np.array(numpyMatrix_data[:,:]).astype('int')
    x=np.delete(x, [0], axis=1)
    y=np.array(numpyMatrix_data[:,0]).astype('int')
    return x,y
    
########## Apply SVD ##########################
def svd(x_train,x_test):    
    svd = TruncatedSVD(n_components=200, n_iter=7, random_state=42)
    x_train_reduced=svd.fit_transform(x_train)
    x_test_reduced=svd.fit_transform(x_test)
    return x_train_reduced,x_test_reduced
    
########Model Fit and Predict########
def modelFitandPredict(model,x_train,y_train,x_test):
    x_train,x_test=svd(x_train,x_test)
    model.fit(x_train,y_train)    
    y_test= model.predict(x_test)
    print(model)
    print(confusionMatrix(y_actual,y_test))
    print("Cohen's kappa ",cohen_kappa_score(y_actual,y_test))
#    print(roc_auc_score(y_actual,y_test))
    return y_test

############## Main Code Starts #################
reading_directory='C:/Nishant/Personal/Assignment/Final/Scores'
writing_directory='C:/Nishant/Personal/Assignment/Final/Scores'

data=readCSVFile(reading_directory,"SampleDataTDMSVD.csv")
data.fillna(0)
data['Level1_Categorical']=data['Level1']
data=labelEncoding(data,['Level1'])
train,test=divideTrainTest(data)

#train1=train.drop(['S.No','Text','Level2','CleanedText','CharacterisedText','CharacterisedNumber','Level1_Categorical'],axis=1)
#test1=test.drop(['S.No','Text','Level2','CleanedText','CharacterisedText','CharacterisedNumber','Level1_Categorical'],axis=1)

train1=train.drop(['S.No','Text','Level2','CharacterisedText','Level1_Categorical'],axis=1)
test1=test.drop(['S.No','Text','Level2','CharacterisedText','Level1_Categorical'],axis=1)

##### BuidldingNumpyMatrix Training and Test ##############
x_train,y_train=numpyMatrix(train1)
x_test,y_actual=numpyMatrix(test1)

######### Bernoulli Naive Bayesian Model Fitting ###########
#model = BernoulliNB()
#y_test_naive=modelFitandPredict(model,x_train,y_train,x_test)

######### DecisionTree Model Fitting ###########
#model = tree.DecisionTreeClassifier(criterion='gini')
#y_test_decision=modelFitandPredict(model,x_train,y_train,x_test)

######### Random Forest Model Fitting ###########
model= RandomForestClassifier(n_estimators=1000)
y_test_random=modelFitandPredict(model,x_train,y_train,x_test)

######### XGBoost Model Fitting ###########
#model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
#y_test_XGBoost=modelFitandPredict(model,x_train,y_train,x_test)

######## Writing Test File ###########
test['PredictedLevel1']=y_test_random
writeCSVFile(train,writing_directory,"Training.csv")
writeCSVFile(test,writing_directory,"Test.csv")


