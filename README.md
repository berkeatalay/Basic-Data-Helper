# Basic-Data-Helper

This is a package that is made by me to gain automated insights and create basic models.

It is Process_Classes.py, you can download it and use it with your Data Science projects. Examples on how to use it is present in Classification_Example.ipynb and Regression_Example.ipynb

## Features

### information(path, file name with extention)

Information is a class for gaining most basic insights. You can drop tables, transform your categorical data to dummies. it can be used as information("data/","hmelq.csv")

It has the following functions:
#### getInfo(load = True, method = "")
  * load is True by default so if you dont want to change your data enter it as False
  * Method is currenty only have "drop" function and it drops rows with missing values
#### catInfo()
  * Describes first rows with object class
#### dropColumn(column_list = [])
  * Dropping columns your data
#### catToDummy(self,column_names = []):
  * If column names are given by user, this functions encode those columns and drops original
  * If column names are not given, it encodes all columns with object class
#### chooseYourTarget(column_name, hist=False ):
  * Returns X and Y from your date to use in models
  * When hist is True, this function also creates histograms of all data without your target

### dataClassification()

This class has basic machine learning models to create a first idea on your models

#### split(rs=0):
  * This function splits data to 30% test, 70% train and save it as X_train, X_test, y_train , y_test with given random state
#### defaultProcesses()
  * This function uses the following models with default parameters
    * Logistic Regression
    * Decision Tree
    * Random Forest
    * MLP
    * SVC
    * Gradient Boosting
    * XGBoosting
    * CatBoost
    * KNeighbors
  * After models are used it crates a plot that is ordered by f1 score and shows acc and f1 score. Example can be seen in Classification_Example.ipynb
#### grid(method, params = {},ver=2)
  * You can use your own parameters and method to use any of the models above
  * methods needs to be one of 'lr','dt','rf','mlpc','svm','gbm','xgb','lgbm','cat','knn'. 
  * This prints you a classification report and gives you best model parameters for selected model and parameters. Predetermined parameters can be seen in file

### gridAll(clf_list=[])
  * This methods is for using grid search on multiple models. 
  * As an extra feature it also gives tunes acc,f1 score versus normal ones for each model that is entered.
  * IF clf_list is empty then it perfoms it on all of model above.
