import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,cohen_kappa_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier 
#from sklearn import svm

########## Read CSV File ############
def readCSVFile(reading_directory,filename):
    data=pd.read_csv(reading_directory+'/'+filename)
    return data

########## Write Datframe Into CSVFile ############
def writeCSVFile(data,writing_directory,filename):
    data.to_csv(writing_directory+'/'+filename,index=False)   

def divideTrainTest(data):
    train=data.sample(frac=0.7,random_state=200)
    test=data.drop(train.index)
    return (train,test)
    
######Converting Categorical To Numerical##############
def labelEncoding(data,var_mod):
    le = LabelEncoder()   
    for i in var_mod:
        data[i] = le.fit_transform(data[i])
    return data
    
######Building Confsuion Matrix##############
def confusionMatrix(Y_actual,Y_test):
    return(confusion_matrix(Y_actual, Y_test))

##### Building Numpy Mtarix ##########
def numpyMatrix(data):
    numpyMatrix_data = data.as_matrix()
    x=np.array(numpyMatrix_data[:,:]).astype('int')
    x=np.delete(x, [0], axis=1)
    y=np.array(numpyMatrix_data[:,0]).astype('int')
    return x,y
    
########Model Fit and Predict########
def modelFitandPredict(model,x_train,y_train,x_test)    :
    model.fit(x_train,y_train)
    y_test= model.predict(x_test)
    print(cohen_kappa_score(y_actual,y_test))
    return y_test

############## Main Code Starts #################
reading_directory='C:/Nishant/Personal/Assignment/Final/Scores'
writing_directory='C:/Nishant/Personal/Assignment/Final/Scores'

train=readCSVFile(reading_directory,"TrainingRandomUsingScores.csv")
train['Level2_Categorical']=train['Level2']
train=labelEncoding(train,['Level2'])

test=readCSVFile(reading_directory,"TestRandomLevelUsingScores.csv")
test['Level2_Categorical']=test['Level2']
test=labelEncoding(test,['Level2'])

Level1_categories=train['Level1'].unique().tolist()

sum=0
result = pd.DataFrame()
for i in Level1_categories:
#    if(sum>5):
#        break
#    sum+=1
    train_small=train.loc[train['Level1']==i]
    test_small=test.loc[test['PredictedLevel1']==i]
    
    train1=train_small.drop(['S.No','Text','Level1','CleanedText','CharacterisedText','Level2_Categorical','CharacterisedNumber','Level1_Categorical'],axis=1)
    test1=test_small.drop(['S.No','Text','Level1','CleanedText','CharacterisedText','Level2_Categorical','CharacterisedNumber','Level1_Categorical','PredictedLevel1'],axis=1)
    
    ##### BuidldingNumpyMatrix Training and Test ##############
    x_train,y_train=numpyMatrix(train1)
    x_test,y_actual=numpyMatrix(test1)
    
    ######## Bernoulli Naive Bayesian Model Fitting ###########
    if len(x_test)!=0:
#        model = BernoulliNB()    
#        y_test_naive=modelFitandPredict(model,x_train,y_train,x_test)
        print(len(test_small))
    
#        ######## Bernoulli DecisionTree Model Fitting ###########
#        model = tree.DecisionTreeClassifier(criterion='gini')
#        y_test_decision=modelFitandPredict(model,x_train,y_train,x_test)
#        
        ######### Random Forest Model Fitting ###########
        model= RandomForestClassifier(n_estimators=1000)
        y_test_random=modelFitandPredict(model,x_train,y_train,x_test)
#        
#        ######## XGBoost Model Fitting ###########
#        model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
#        y_test_XGBoost=modelFitandPredict(model,x_train,y_train,x_test)
    
        test_small['PredictedLevel2']=y_test_random
        result = result.append(test_small) 
        
    
######## Writing Test File ###########
writeCSVFile(result,writing_directory,"TestDataUsingScores.csv")

