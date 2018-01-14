# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:59:59 2017

@author: Hardik Galiawala
         B00777450
"""

import pandas as pd
import numpy as np
import scipy as sc

from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

# creating object for LogReg
logistic_reg_model = LogisticRegression()
# creating object for Naive Bayes 
naive_bayes = GaussianNB()
#creating object for Decision Tree
decision_tree = DecisionTreeClassifier()
#creating object for Random Forest 
random_forest = RandomForestClassifier()

#creating objects for Random Forest with estimators
random_forest_10 = RandomForestClassifier()
random_forest_20 = RandomForestClassifier(n_estimators = 20)
random_forest_50 = RandomForestClassifier(n_estimators = 50)
random_forest_100 = RandomForestClassifier(n_estimators = 100)

# Method for training Logistic Regression
def train_log_reg(train_x, train_y):    
    return logistic_reg_model.fit(train_x, train_y)

# Method for predicting Logistic Regression values
def predict_log_reg(test_x):
    return logistic_reg_model.predict(test_x)

# Method for training Naive Bayes
def train_naive_bayes(train_x, train_y):
    return naive_bayes.fit(train_x, train_y)

# Method for predicting Naive Bayes
def predict_naive_bayes(test_x):
    return naive_bayes.predict(test_x)

# Method for training Decision Trees
def train_decision_tree(train_x, train_y):
    return decision_tree.fit(train_x, train_y)

# Method for predicting Decision Trees
def predict_decision_tree(test_x):
    return decision_tree.predict(test_x)

# Method for training Random Forest
def train_random_forest(train_x, train_y):    
    return random_forest.fit(train_x, train_y)

# Method for predicting Random Forest values
def predict_random_forest(test_x):
    return random_forest.predict(test_x)

def logReg(train_x, test_x, train_y, test_y, class_name):
    # Logistic Regression
    #Logistic regression trained model
    trained_log_reg_model = train_log_reg(train_x, train_y)
    
    # Logistict regression predicted values for trained data
    predicted_trained = predict_log_reg(train_x)
    
    # Logistict regression predicted values for test data
    predicted_test = predict_log_reg(test_x)
    print('----------------Logistic Regression ' + class_name + ' ---------')
    print('Confusion matrix:-')
    print('Confusion matrix for ' + class_name + ' (Train v/s test)')
    print('Train :')
    print(metrics.confusion_matrix(train_y, predicted_trained))
    print('Test :')
    print(metrics.confusion_matrix(test_y, predicted_test))
    
    print('Accuracy:-')
    print('Accuracy score for ' + class_name + ' (Train v/s test)')
    print(str(metrics.accuracy_score(train_y, predicted_trained)) +' '+ str(metrics.accuracy_score(test_y, predicted_test)))

def naiveBayes(train_x, test_x, train_y, test_y, class_name):
    # Naive Bayes
    # Naive Bayes trained model
    trained_nb_model = train_naive_bayes(train_x, train_y)
    
    # Naive Bayes predicted values for trained data
    predicted_trained = predict_naive_bayes(train_x)
    
    # Naive Bayes predicted values for test data
    predicted_test = predict_naive_bayes(test_x)
    print('---------------- Naive Bayes ' + class_name + ' ---------')
    print('Confusion matrix:-')
    print('Confusion matrix for ' + class_name + ' (Train v/s test)')
    print('Train :')
    print(metrics.confusion_matrix(train_y, predicted_trained))
    print('Test :')
    print(metrics.confusion_matrix(test_y, predicted_test))
    
    print('Accuracy:-')
    print('Accuracy score for ' + class_name + ' (Train v/s test)')
    print(str(metrics.accuracy_score(train_y, predicted_trained)) +' '+ str(metrics.accuracy_score(test_y, predicted_test)))

def decisionTree(train_x, test_x, train_y, test_y, class_name):
    # Decision Tree
    # Decision Tree trained model
    trained_dt_model = train_decision_tree(train_x, train_y)
    
    # Decision Tree predicted values for trained data
    predicted_trained = predict_decision_tree(train_x)
    
    # Decision Tree predicted values for test data
    predicted_test = predict_decision_tree(test_x)
    print('---------------- Decision Tree ' + class_name + ' ---------')
    print('Confusion matrix:-')
    print('Confusion matrix for ' + class_name + ' (Train v/s test)')
    print('Train :')
    print(metrics.confusion_matrix(train_y, predicted_trained))
    print('Test :')
    print(metrics.confusion_matrix(test_y, predicted_test))
    
    print('Accuracy:-')
    print('Accuracy score for ' + class_name + ' (Train v/s test)')
    print(str(metrics.accuracy_score(train_y, predicted_trained)) +' '+ str(metrics.accuracy_score(test_y, predicted_test)))

def randomForest(train_x, test_x, train_y, test_y, class_name):
    # Random Forest
    # Random Forest trained model
    trained_rf_model = train_random_forest(train_x, train_y)
    
    # Random Forest predicted values for trained data
    predicted_trained = predict_random_forest(train_x)
    
    # Random Forest predicted values for test data
    predicted_test = predict_random_forest(test_x)
    print('---------------- Random Forest ' + class_name + ' ---------')
    print('Confusion matrix:-')
    print('Confusion matrix for ' + class_name + ' (Train v/s test)')
    print('Train :')
    print(metrics.confusion_matrix(train_y, predicted_trained))
    print('Test :')
    print(metrics.confusion_matrix(test_y, predicted_test))
    
    print('Accuracy:-')
    print('Accuracy score for ' + class_name + ' (Train v/s test)')
    print(str(metrics.accuracy_score(train_y, predicted_trained)) +' '+ str(metrics.accuracy_score(test_y, predicted_test)))

def main():
    # Reading data from csv file
    raw_data = pd.read_csv('D:/Dalhousie/Term1/ML with BigData/Assignment1/animals.csv')    
    raw_data = raw_data.values
    
    # Converting to array
    features = np.array(raw_data[:,:25])
    animalClass = np.array(raw_data[:,-1:]).ravel()
    
    #Binarizing the classes
    target_deer = [1 if animalClass[i] == 'DEER' else 0 for i in range(len(animalClass))]
    target_elk = [1 if animalClass[i] == 'ELK' else 0 for i in range(len(animalClass))]
    target_cattle = [1 if animalClass[i] == 'CATTLE' else 0 for i in range(len(animalClass))]
    
    # Splitting data into training and testing (70:30)
    train_deer_x, test_deer_x, train_deer_y, test_deer_y = train_test_split(features, target_deer, train_size=0.7)
    train_elk_x, test_elk_x, train_elk_y, test_elk_y = train_test_split(features, target_elk, train_size=0.7)
    train_cattle_x, test_cattle_x, train_cattle_y, test_cattle_y = train_test_split(features, target_cattle, train_size=0.7)
    
    #Logistic Regression
    #calling function to calculate accuracy and confusion matrix for each class
    logReg(train_deer_x, test_deer_x, train_deer_y, test_deer_y, 'DEER')
    logReg(train_elk_x, test_elk_x, train_elk_y, test_elk_y, 'ELK')
    logReg(train_cattle_x, test_cattle_x, train_cattle_y, test_cattle_y, 'CATTLE')
    
    # Naive Bayes
    #calling function to calculate accuracy and confusion matrix for each class
    naiveBayes(train_deer_x, test_deer_x, train_deer_y, test_deer_y, 'DEER')
    naiveBayes(train_elk_x, test_elk_x, train_elk_y, test_elk_y, 'ELK')
    naiveBayes(train_cattle_x, test_cattle_x, train_cattle_y, test_cattle_y, 'CATTLE')
    
    # Decision Tree
    #calling function to calculate accuracy and confusion matrix for each class
    decisionTree(train_deer_x, test_deer_x, train_deer_y, test_deer_y, 'DEER')
    decisionTree(train_elk_x, test_elk_x, train_elk_y, test_elk_y, 'ELK')
    decisionTree(train_cattle_x, test_cattle_x, train_cattle_y, test_cattle_y, 'CATTLE')
    
    # Random Forest
    #calling function to calculate accuracy and confusion matrix for each class
    randomForest(train_deer_x, test_deer_x, train_deer_y, test_deer_y, 'DEER')
    randomForest(train_elk_x, test_elk_x, train_elk_y, test_elk_y, 'ELK')
    randomForest(train_cattle_x, test_cattle_x, train_cattle_y, test_cattle_y, 'CATTLE')

    
    # 10 fold Cross validation for logistic regression done on full data without splitting
    
    # Class - DEER
    cross_val_lr_deer = cross_val_score(logistic_reg_model, features, target_deer, cv=10, scoring = 'accuracy')
    print("Logistict Regression - Deer")
    print("Mean Accuracy : " + str(cross_val_lr_deer.mean()) + "Standard Deviation : " + str(np.std(cross_val_lr_deer, axis = 0)))
    
    cross_val_nb_deer = cross_val_score(naive_bayes, features, target_deer, cv=10, scoring = 'accuracy')
    print("Naive Bayes - Deer")
    print("Mean Accuracy : " + str(cross_val_nb_deer.mean()) + "Standard Deviation : " + str(np.std(cross_val_nb_deer, axis = 0)))
    
    cross_val_dt_deer = cross_val_score(decision_tree, features, target_deer, cv=10, scoring = 'accuracy')
    print("Decision Tree - Deer")
    print("Mean Accuracy : " + str(cross_val_dt_deer.mean()) + "Standard Deviation : " + str(np.std(cross_val_dt_deer, axis = 0)))
    
    cross_val_rf_deer = cross_val_score(random_forest, features, target_deer, cv=10, scoring = 'accuracy')
    print("Random Forest - Deer")
    print("Mean Accuracy : " + str(cross_val_rf_deer.mean()) + "Standard Deviation : " + str(np.std(cross_val_rf_deer, axis = 0)))
    
    # Class - ELK
    cross_val_lr_elk = cross_val_score(logistic_reg_model, features, target_elk, cv=10, scoring = 'accuracy')
    print("Logistict regression - Elk")
    print("Mean Accuracy : " + str(cross_val_lr_elk.mean()) + "Standard Deviation : " + str(np.std(cross_val_lr_elk, axis = 0)))

    cross_val_nb_elk = cross_val_score(naive_bayes, features, target_elk, cv=10, scoring = 'accuracy')
    print("Naive Bayes - Elk")
    print("Mean Accuracy : " + str(cross_val_nb_elk.mean()) + "Standar Deviation : " + str(np.std(cross_val_nb_elk, axis = 0)))
    
    cross_val_dt_elk = cross_val_score(decision_tree, features, target_elk, cv=10, scoring = 'accuracy')
    print("Decision Tree - Elk")
    print("Mean Accuracy : " + str(cross_val_dt_elk.mean()) + "Standar Deviation : " + str(np.std(cross_val_dt_elk, axis = 0)))
    
    cross_val_rf_elk = cross_val_score(random_forest, features, target_elk, cv=10, scoring = 'accuracy')
    print("Random Forest - Elk")
    print("Mean Accuracy : " + str(cross_val_rf_elk.mean()) + "Standar Deviation : " + str(np.std(cross_val_rf_elk, axis = 0)))
    
    # Class - CATTLE
    cross_val_lr_cattle = cross_val_score(logistic_reg_model, features, target_cattle, cv=10, scoring = 'accuracy')
    print("Logistict regression - Cattle")
    print("Mean Accuracy : " + str(cross_val_lr_cattle.mean()) + "Standar Deviation : " + str(np.std(cross_val_lr_cattle, axis = 0)))
    
    cross_val_nb_cattle = cross_val_score(naive_bayes, features, target_cattle, cv=10, scoring = 'accuracy')
    print("Naive Bayes - Cattle")
    print("Mean Accuracy : " + str(cross_val_nb_cattle.mean()) + "Standar Deviation : " + str(np.std(cross_val_nb_cattle, axis = 0)))
    
    cross_val_dt_cattle = cross_val_score(decision_tree, features, target_cattle, cv=10, scoring = 'accuracy')
    print("Decision Tree : Cattle")
    print("Mean Accuracy : " + str(cross_val_dt_cattle.mean()) + "Standar Deviation : " + str(np.std(cross_val_dt_cattle, axis = 0)))
    
    cross_val_rf_cattle = cross_val_score(random_forest, features, target_cattle, cv=10, scoring = 'accuracy')
    print("Random Forest - Cattle")
    print("Mean Accuracy : " + str(cross_val_rf_cattle.mean()) + "Standar Deviation : " + str(np.std(cross_val_rf_cattle, axis = 0)))
    
    
    # Concatenating accuracy of different classes into a single array per classifier
    cross_val_lr_agg = np.concatenate((cross_val_lr_deer, cross_val_lr_elk, cross_val_lr_cattle), axis = 0)
    cross_val_nb_agg = np.concatenate((cross_val_nb_deer, cross_val_nb_elk, cross_val_nb_cattle), axis = 0)
    cross_val_dt_agg = np.concatenate((cross_val_dt_deer, cross_val_dt_elk, cross_val_dt_cattle), axis = 0)
    cross_val_rf_agg = np.concatenate((cross_val_rf_deer, cross_val_rf_elk, cross_val_rf_cattle), axis = 0)
    
    # Aggregated mean value of 10-fold cross validation
    print("Aggregated Mean and Standard Deviation [Accuracy]")
    cross_val_lr_mean, cross_val_lr_std = cross_val_lr_agg.mean(), np.std(cross_val_lr_agg, axis = 0)
    cross_val_nb_mean, cross_val_nb_std = cross_val_nb_agg.mean(), np.std(cross_val_nb_agg, axis = 0)
    cross_val_dt_mean, cross_val_dt_std = cross_val_dt_agg.mean(), np.std(cross_val_dt_agg, axis = 0)
    cross_val_rf_mean, cross_val_rf_std = cross_val_rf_agg.mean(), np.std(cross_val_rf_agg, axis = 0)
    print("Logistic Regression - Mean and SD of accuracy of 10 crossfolds")
    print(str(cross_val_lr_mean) + " and " + str(cross_val_lr_std))
    print("Naive Bayes - Mean and SD of accuracy of 10 crossfolds")
    print(str(cross_val_nb_mean) + " and " + str(cross_val_nb_std))
    print("Decision Tree - Mean and SD of accuracy of 10 crossfolds")
    print(str(cross_val_dt_mean) + " and " + str(cross_val_dt_std))
    print("Random Forest - Mean and SD of accuracy of 10 crossfolds")
    print(str(cross_val_rf_mean) + " and " + str(cross_val_rf_std))
    
    # Calculating p-values with Student's t-test 
    # Random Forest v/s other classifiers
    stats_lr, p_val_lr = sc.stats.ttest_rel(cross_val_rf_agg, cross_val_lr_agg, axis = 0)
    stats_nb, p_val_nb = sc.stats.ttest_rel(cross_val_rf_agg, cross_val_nb_agg, axis = 0)
    stats_dt, p_val_dt = sc.stats.ttest_rel(cross_val_rf_agg, cross_val_dt_agg, axis = 0)
    
    if(p_val_lr <= 0.05):
        print("Good P-value, hence data did not occur by chance for Logistic regression")
        
    else:
        print("Bad P-value")
        
    if(p_val_nb <= 0.05):
        print("Good P-value, hence data did not occur by chance for Naive Bayes")
        
    else:
        print("Bad P-value")
    if(p_val_dt <= 0.05):
        print("Good P-value, hence data did not occur by chance for Decision Tree")
        
    else:
        print("Bad P-value")
        
    # Cross validation score
    # 10 fold Cross validation for random forest done on full data without splitting
    # Estimators = 10
    # This is done as a formality and is not needed as we have done it in the above part
    # Cross Validation (Random forest) per Animal Class
    cross_val_rf_deer_10 = cross_val_score(random_forest_10, features, target_deer, cv=10, scoring = 'accuracy')
    cross_val_rf_elk_10 = cross_val_score(random_forest_10, features, target_elk, cv=10, scoring = 'accuracy')
    cross_val_rf_cattle_10 = cross_val_score(random_forest_10, features, target_cattle, cv=10, scoring = 'accuracy')
    
    # Aggregating accuracies of all Animal Classes into a single array
    cross_val_rf10_agg = np.concatenate((cross_val_rf_deer_10, cross_val_rf_elk_10, cross_val_rf_cattle_10), axis = 0)
    
    # Calculating Mean and Standard Deviation
    cv_rf_10_mean, cv_rf_10_std = cross_val_rf10_agg.mean(), np.std(cross_val_rf10_agg, axis = 0)
    print("Random Forest 10 Estimators")
    print("Mean : ", cv_rf_10_mean )
    print("Standard Deviation : ", cv_rf_10_std)
    
    # Cross validation score
    # 10 fold Cross validation for random forest done on full data without splitting
    # Estimators = 20
    
    # Cross Validation (Random forest) per Animal Class
    cross_val_rf_deer_20 = cross_val_score(random_forest_20, features, target_deer, cv=10, scoring = 'accuracy')
    cross_val_rf_elk_20 = cross_val_score(random_forest_20, features, target_elk, cv=10, scoring = 'accuracy')
    cross_val_rf_cattle_20 = cross_val_score(random_forest_20, features, target_cattle, cv=10, scoring = 'accuracy')
    
    # Aggregating accuracies of all Animal Classes into a single array
    cross_val_rf20_agg = np.concatenate((cross_val_rf_deer_20, cross_val_rf_elk_20, cross_val_rf_cattle_20), axis = 0)
    
    # Calculating Mean and Standard Deviation
    cv_rf_20_mean, cv_rf_20_std = cross_val_rf20_agg.mean(), np.std(cross_val_rf20_agg, axis = 0)
    print("Random Forest 20 Estimators")
    print("Mean : ", cv_rf_20_mean )
    print("Standard Deviation : ", cv_rf_20_std)
    print (cv_rf_20_mean)
    
    # Cross validation score
    # 10 fold Cross validation for random forest done on full data without splitting
    # Estimators = 50
    
    # Cross Validation (Random forest) per Animal Class
    cross_val_rf_deer_50 = cross_val_score(random_forest_50, features, target_deer, cv=10, scoring = 'accuracy')
    cross_val_rf_elk_50 = cross_val_score(random_forest_50, features, target_elk, cv=10, scoring = 'accuracy')
    cross_val_rf_cattle_50 = cross_val_score(random_forest_50, features, target_cattle, cv=10, scoring = 'accuracy')
    
    # Aggregating accuracies of all Animal Classes into a single array
    cross_val_rf50_agg = np.concatenate((cross_val_rf_deer_50, cross_val_rf_elk_50, cross_val_rf_cattle_50), axis = 0)
    
    # Calculating Mean and Standard Deviation
    cv_rf_50_mean, cv_rf_50_std = cross_val_rf50_agg.mean(), np.std(cross_val_rf50_agg, axis = 0)
    print("Random Forest 50 Estimators")
    print("Mean : ", cv_rf_50_mean )
    print("Standard Deviation : ", cv_rf_50_std)
    
    # Cross validation score
    # 10 fold Cross validation for random forest done on full data without splitting
    # Estimators = 100
    
    # Cross Validation (Random forest) per Animal Class
    cross_val_rf_deer_100 = cross_val_score(random_forest_100, features, target_deer, cv=10, scoring = 'accuracy')
    cross_val_rf_elk_100 = cross_val_score(random_forest_100, features, target_elk, cv=10, scoring = 'accuracy')
    cross_val_rf_cattle_100 = cross_val_score(random_forest_100, features, target_cattle, cv=10, scoring = 'accuracy')
    
    # Aggregating accuracies of all Animal Classes into a single array
    cross_val_rf100_agg = np.concatenate((cross_val_rf_deer_100, cross_val_rf_elk_100, cross_val_rf_cattle_100), axis = 0)
    
    # Calculating Mean and Standard Deviation
    cv_rf_100_mean, cv_rf_100_std = cross_val_rf100_agg.mean(), np.std(cross_val_rf100_agg, axis = 0)
    print("Random Forest 100 Estimators")
    print("Mean : ", cv_rf_100_mean )
    print("Standard Deviation : ", cv_rf_100_std)
    
    # Calculating p-values
    # Comparing Random Forest(20 - Estimators) v/s Other classifiesr (LR, NB and DT)
    stats_lr_20, p_val_lr_20 = sc.stats.ttest_rel(cross_val_rf20_agg, cross_val_lr_agg, axis = 0)
    stats_nb_20, p_val_nb_20 = sc.stats.ttest_rel(cross_val_rf20_agg, cross_val_nb_agg, axis = 0)
    stats_dt_20, p_val_dt_20 = sc.stats.ttest_rel(cross_val_rf20_agg, cross_val_dt_agg, axis = 0)
    
    if(p_val_lr_20 <= 0.05):
        print("Good P-value, hence data did not occur by chance for Logistic regression")
        print(p_val_lr_20)
    else:
        print("Bad P-value")
        print(p_val_lr_20)
    if(p_val_nb_20 <= 0.05):
        print("Good P-value, hence data did not occur by chance for Naive Bayes")
        print(p_val_nb_20)
    else:
        print("Bad P-value")
    if(p_val_dt_20 <= 0.05):
        print("Good P-value, hence data did not occur by chance for Decision Tree")
        print(p_val_dt_20)
    else:
        print("Bad P-value")

main()