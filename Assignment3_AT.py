#In functions capital letter X,Y for training, small letter x,y for test
#IN GRAPHS blue color is train data, red color is test data, black is error
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt

def decision_tree_tuning(X,Y,x,y):
    '''
    This function fits the training data X(features) and Y(output) in a 
    decision tree and returns the accuracy score with best hyperparameters 
    max depth of tree and min samples in every leaf. (X=trainX,Y=trainY,x=TestX,y=TestY)
    '''
    #depth optimisation
    for i in range(1,60):
        tree = DecisionTreeClassifier(criterion='gini',max_depth=i)
        tree.fit(X,Y)
        Y_pred=tree.predict(X) #training prediction
        y_pred=tree.predict(x) #test prediction
        #depth vs AUC plot for training data and test data
        #training data AUC calculation
        fpr,tpr,th=roc_curve(Y,Y_pred)
        a_train = auc(fpr, tpr)
        #test data AUC calculation
        fpr,tpr,th=roc_curve(y,y_pred)
        a_test = auc(fpr,tpr)
        #test train error calculation
        error=a_train-a_test
        plt.scatter(i,a_train, color='blue',label='Train data')
        plt.scatter(i,a_test,color='red', label='Test data')
       # plt.scatter(i,error,color='brown')
    plt.title("Decision tree: Depth vs AUC")
    plt.xlabel("Depth")
    plt.ylabel("AUC")
    plt.show()

    #min samples per leaf optimisation
    for i in range(1,60):
        tree = DecisionTreeClassifier(criterion='gini',min_samples_leaf=i)
        tree.fit(X,Y)
        Y_pred=tree.predict(X) #training prediction
        y_pred=tree.predict(x) #test prediction
        #min_sample vs AUC plot for training data and test data
        #training data AUC calculation
        fpr,tpr,th=roc_curve(Y,Y_pred)
        a_train = auc(fpr, tpr)
        #test data AUC calculation
        fpr,tpr,th=roc_curve(y,y_pred)
        a_test = auc(fpr,tpr)
        #test train error calculation
        error=a_train-a_test
        plt.scatter(i,a_train, color='blue',label='Train data')
        plt.scatter(i,a_test,color='red', label='Test data')
       # plt.scatter(i,error,color='brown')
    plt.title("Decision tree: Min_samples_leaf vs AUC")
    plt.xlabel("Min_samples_leaf")
    plt.ylabel("AUC")
    plt.show()
    #final optimisation result
    t0=time.time()
    tree = DecisionTreeClassifier(criterion='gini',max_depth=4,min_samples_leaf=10)
    tree.fit(X,Y)
    tf=time.time()
    return accuracy_score(y,tree.predict(x)), tf-t0

'''    #5 fold cross validation for different depths
    for i in range(1,20):
        tree = DecisionTreeClassifier(criterion='gini',max_depth=i) 
        score = (cross_val_score(estimator=tree, X=X, y=Y, cv=5)).mean()
        if score>temp_score:
            TREE=tree
            I=i
            temp_score=score
        print(score,i)
    TREE.fit(X,Y)
'''
tf=time.time()
    

def random_forest_tuning(X,Y,x,y):
    '''
    This function fits the training data X(features) and Y(output) in a 
    random forest and returns the accuracy score with best hyperparameters 
    number of estimators and min samples in every leaf. 
    (X=trainX,Y=trainY,x=TestX,y=TestY)
    '''
    #nno. of trees optimisation
    for i in range(10,100):
        forest = RandomForestClassifier(criterion='gini',n_estimators=i)
        forest.fit(X,Y)
        Y_pred=forest.predict(X) #training prediction
        y_pred=forest.predict(x) #test prediction
        #n_estimators vs AUC plot for training data and test data
        #training data AUC calculation
        fpr,tpr,th=roc_curve(Y,Y_pred)
        a_train = auc(fpr, tpr)
        #test data AUC calculation
        fpr,tpr,th=roc_curve(y,y_pred)
        a_test = auc(fpr,tpr)
        #test train error calculation
        error=a_train-a_test
        plt.scatter(i,a_train, color='blue',label='Train data')
        plt.scatter(i,a_test,color='red', label='Test data')
       # plt.scatter(i,error,color='brown')
    plt.title("Random Forest: n_estimators vs AUC")
    plt.xlabel("No. of trees")
    plt.ylabel("AUC")
    plt.show()

    #max features optimisation
    for i in range(3,30):
        forest = DecisionTreeClassifier(criterion='gini',min_samples_leaf=i)
        forest.fit(X,Y)
        Y_pred=forest.predict(X) #training prediction
        y_pred=forest.predict(x) #test prediction
        #max_feature vs AUC plot for training data and test data
        #training data AUC calculation
        fpr,tpr,th=roc_curve(Y,Y_pred)
        a_train = auc(fpr, tpr)
        #test data AUC calculation
        fpr,tpr,th=roc_curve(y,y_pred)
        a_test = auc(fpr,tpr)
        #test train error calculation
        error=a_train-a_test
        plt.scatter(i,a_train, color='blue',label='Train data')
        plt.scatter(i,a_test,color='red', label='Test data')
       # plt.scatter(i,error,color='brown')
    plt.title("Random forest: max_features vs AUC")
    plt.xlabel("max features")
    plt.ylabel("AUC")
    plt.show()
    #final optimisation result
    t0=time.time()
    forest = RandomForestClassifier(criterion='gini',n_estimators=100,max_features=10)
    forest.fit(X,Y)
    tf=time.time()
    return accuracy_score(y,forest.predict(x)), tf-t0

def svm_tuning(X,Y,x,y):
    '''
    This function fits the training data X(features) and Y(output) in a 
    SVM and returns the accuracy score with best hyperparameters 
    C and gamma. ALSO KERNEL USED IS GAUSSIAN TO PROVIDE A NON LINEAR BASIC
    FITTING, AFTER THAT GAMMA IS USED TO INCREASE THE DEGREE OF FITTING.
    (X=trainX,Y=trainY,x=TestX,y=TestY)
    '''
    #gamma optimisation
    for i in range(-2,3):
        svm_model = svm.SVC(kernel='rbf',gamma=10**i)
        svm_model.fit(X,Y)
        Y_pred=svm_model.predict(X) #training prediction
        y_pred=svm_model.predict(x) #test prediction
        #gamma vs AUC plot for training data and test data
        #training data AUC calculation
        fpr,tpr,th=roc_curve(Y,Y_pred)
        a_train = auc(fpr, tpr)
        #test data AUC calculation
        fpr,tpr,th=roc_curve(y,y_pred)
        a_test = auc(fpr,tpr)
        #test train error calculation
        error=a_train-a_test
        plt.scatter(i,a_train, color='blue',label='Train data')
        plt.scatter(i,a_test,color='red', label='Test data')
       # plt.scatter(i,error,color='brown')
    plt.title("SVM: log(gamma) vs AUC")
    plt.xlabel("log(gamma)")
    plt.ylabel("AUC")
    plt.show()

    #C optimisation
    for i in range(1,10):
        svm_model = svm.SVC(kernel='rbf',C=10*i,gamma='auto')
        svm_model.fit(X,Y)
        Y_pred=svm_model.predict(X) #training prediction
        y_pred=svm_model.predict(x) #test prediction
        #C vs AUC plot for training data and test data
        #training data AUC calculation
        fpr,tpr,th=roc_curve(Y,Y_pred)
        a_train = auc(fpr, tpr)
        #test data AUC calculation
        fpr,tpr,th=roc_curve(y,y_pred)
        a_test = auc(fpr,tpr)
        #test train error calculation
        error=a_train-a_test
        plt.scatter(10*i,a_train, color='blue',label='Train data')
        plt.scatter(10*i,a_test,color='red', label='Test data')
       # plt.scatter(i,error,color='brown')
    plt.title("SVM: C vs AUC")
    plt.xlabel("C")
    plt.ylabel("AUC")
    plt.show()
    #final optimisation result
    t0=time.time()
    svm_model = svm.SVC(kernel='rbf',C=30,gamma=1)
    svm_model.fit(X,Y)
    tf=time.time()    
    return accuracy_score(y,svm_model.predict(x)), tf-t0


data=pd.read_csv('/home/abhinav/Desktop/Assignment3_AT/sonar.all-data', delimiter=',', header=None)


df=data.replace(to_replace =("R","M"), value =(1,0))
data.update(df)
#data preprocessing and visualisation
print('Total no. of observations:',data.shape[0])
print("% of Rock rebounds:",len(data[data.iloc[:,60]==1])*100/data.shape[0])
print("% of Metal rebounds:",len(data[data.iloc[:,60]==0])*100/data.shape[0])
print('No. of null values:', (data.count()-data.isnull().count()).sum())
print("\n")
#feature scaling
data = (data - data.min())/(data.max()-data.min())

#train test split
X_train,X_test,Y_train,Y_test=train_test_split(data.iloc[:,:-1],data.iloc[:,60],test_size=0.2, random_state=100)
Y_train = np.asarray(Y_train, dtype=np.float64)
Y_test=np.asarray(Y_test, dtype=np.float64)
DT_acc,DT_time=decision_tree_tuning(X_train,Y_train,X_test,Y_test)
RF_acc,RF_time=random_forest_tuning(X_train,Y_train,X_test,Y_test)
SVM_acc,SVM_time=svm_tuning(X_train,Y_train,X_test,Y_test)

print("Decision tree accuracy %: ",round(DT_acc,4)*100,"Time: ", round(DT_time,3))
print("Random forest accuracy %: ", round(RF_acc,4)*100,"Time: ",round(RF_time,3))
print("SVM accuracy %: ",round(SVM_acc,4)*100,"Time: ", round(SVM_time,3))




