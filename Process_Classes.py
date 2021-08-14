
import pandas as pd
import numpy as np   
from IPython.display import display


class information:    
    def __init__(self, path, file, data= pd.DataFrame(),catlist = [], catdata= pd.DataFrame()):
        self.path = path
        self.file_name = file
        self.data = data
        self.catlist = catlist
        print("Class object initialized") 
       
       
    def _load_data_info(self):
        #kendime not: file name için . dan ayır sondakini al eğer csv ise read csv değilse ne olduğuna göre güncelle
        typ = self.file_name.split(".")[-1]
        
        if typ == "csv":
            data = pd.read_csv(self.path + self.file_name)
            
        elif typ == "xlsx":
            data = pd.read_excel(self.path + self.file_name)
        
        else:
            print("This file type is not supported. Currently only xlsx and csv is supported. Please open a pr for new data type")
        
        print(self.file_name +" is loaded")
        return data
    
    def getInfo(self,load = True, method = ""):
        """
        Prints basic information about raw data and collect column names with object type.
        Load is True by default
        Method currently only takes 'drop'. 'Drop' is dropping rows with missing values.
        """

        if load:
            data = self._load_data_info()
            self.data = data
            
        if method == "drop":
            print('Row count before removal: ' + str(len(self.data)))
            self.data = self.data.dropna()
            print('Row count after removal: ' + str(len(self.data)))
            
        display(self.data.head())
        print()
        display(self.data.info())
        print()
        display(self.data.describe().T)
        
        print("Columns names with object type are saved in .catlist if you want to turn them into dummies use cattodummy")
        self.catlist = list(self.data.select_dtypes(include=['object']).columns)
        
    
    def catInfo(self):
        display(self.data[self.catlist].head())
        
    def dropColumn(self, column_list = []):
        for i in column_list:
            if i in self.catlist:
                self.catlist.remove(i)
            self.data = self.data.drop(i,axis = 1)
            
 
    def catToDummy(self,column_names = []):
        #if user does not give spesific columns to transform it takes columns with object type
        if column_names == []:
            column_names = self.catlist
        
        print ("Looking for: " + str(column_names))
        
        if (set(column_names).issubset(self.catlist) & set(column_names).issubset(self.data)): #checks if column_names in catdata to prevent errors
            
            self.catdata = pd.get_dummies(self.data, columns=column_names, drop_first= True )    
            print("Dummies are combined with data and saved as .catdata")
            display(self.catdata.head())
        else:
            #error message for columns
            print("Columns do not exist in data you may already transfer them to dummies, please check:")
            display(self.data.head())
        

    def chooseYourTarget (self,column_name, hist=False ):  #returns X , y depending on your selection of column name
        
        #self.catdata[column_name].value_counts().plot.barh()
        x = self.catdata.drop(column_name, axis=1)
        display(x.head())
        if hist:
            x.hist(figsize=(20, 30));

        return self.catdata[column_name], x
 
            
        
        
class dataClassification():
    
    def __init__(self,x,y):
        self.x = x
        self.y = y
        print("Classification class")
        
    def split(self, rs=0): #datayı split ettiğin fonction (Kendime Not : Bunu direk init içerisinde çağır)
        from sklearn.model_selection import train_test_split
        if rs:
            X_train, X_test, y_train , y_test = train_test_split(self.x,self.y,test_size = 0.3, random_state=rs)
        else:
            X_train, X_test, y_train , y_test = train_test_split(self.x,self.y,test_size = 0.3)
            
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def defaultProcesses(self): #Bildiğimiz classification yöntemlerini hiçbir hyperparametre değiştirmeden uyguluyor
        
        from warnings import filterwarnings


        filterwarnings('ignore')
        from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
        from sklearn.neighbors import KNeighborsClassifier

        from sklearn.metrics import classification_report, accuracy_score, f1_score
        
        self.acc = []
        self.f1 = []
        acc_colmns  =['lr','dt','rf','mlpc','svm','gbm','xgb','lgbm','cat','knn']


        lr = LogisticRegression()
        lr.fit(self.X_train,self.y_train)
        y_pred = lr.predict(self.X_test)
        print('Results for default logistic regression')
        print(classification_report(self.y_test,y_pred))
        self.acc.append(accuracy_score(self.y_test,y_pred))
        self.f1.append(f1_score(self.y_test,y_pred))
               
        dtc = DecisionTreeClassifier()
        dtc.fit(self.X_train,self.y_train)
        y_pred = dtc.predict(self.X_test)
        print('Results for default decision tree')
        print(classification_report(self.y_test,y_pred))
        self.acc.append(accuracy_score(self.y_test, y_pred))
        self.f1.append(f1_score(self.y_test, y_pred))

        rfc = RandomForestClassifier()
        rfc.fit(self.X_train,self.y_train)
        y_pred = rfc.predict(self.X_test)
        print('Results for default random forest')
        print(classification_report(self.y_test,y_pred))
        self.acc.append(accuracy_score(self.y_test, y_pred))
        self.f1.append(f1_score(self.y_test, y_pred))
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        X_train_scaled = scaler.transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
                
        mlpc = MLPClassifier()
        mlpc.fit(X_train_scaled,self.y_train)
        y_pred = mlpc.predict(X_test_scaled)
        print('Results for default MLP')
        print(classification_report(self.y_test,y_pred))
        self.acc.append(accuracy_score(self.y_test,y_pred))
        self.f1.append(f1_score(self.y_test,y_pred))


        svc = SVC()
        svc.fit(self.X_train, self.y_train)
        y_pred = svc.predict(self.X_test)
        print('Results for default SVC')
        print(classification_report(self.y_test, y_pred))
        self.acc.append(accuracy_score(self.y_test, y_pred))
        self.f1.append(f1_score(self.y_test, y_pred))

        gbc = GradientBoostingClassifier()
        gbc.fit(self.X_train, self.y_train)
        y_pred = gbc.predict(self.X_test)
        print('Results for default Gradient Boosting Classifier')
        print(classification_report(self.y_test, y_pred))
        self.acc.append(accuracy_score(self.y_test, y_pred))
        self.f1.append(f1_score(self.y_test, y_pred))

        xgb = XGBClassifier().fit(self.X_train, self.y_train)
        y_pred = xgb.predict(self.X_test)
        print('Results for default XGBoost Classifier')
        print(classification_report(self.y_test, y_pred))
        self.acc.append(accuracy_score(self.y_test, y_pred))
        self.f1.append(f1_score(self.y_test, y_pred))

        lgbm = LGBMClassifier().fit(self.X_train, self.y_train)
        print('Results for default LGBMClassifier')
        y_pred = lgbm.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        self.acc.append(accuracy_score(self.y_test, y_pred))
        self.f1.append(f1_score(self.y_test, y_pred))

        cat = CatBoostClassifier(silent=True).fit(self.X_train, self.y_train)

        print('Results for default CatBoost')
        y_pred = cat.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        self.acc.append(accuracy_score(self.y_test,y_pred))
        self.f1.append(f1_score(self.y_test,y_pred))

        knn = KNeighborsClassifier().fit(self.X_train, self.y_train)
        y_pred = knn.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        self.acc.append(accuracy_score(self.y_test, y_pred))
        self.f1.append(f1_score(self.y_test, y_pred))



        self.acc = [i * 100 for i in self.acc]
        self.f1 = [ i* 100 for i in self.f1]
        


        zippedList = list(zip(self.acc,self.f1))
        dfObj = pd.DataFrame(zippedList, columns=['Accuracy', 'F1'], index=acc_colmns)
        dfObj.sort_values(by='F1').plot(kind = 'bar',figsize =(12,12),grid =True)

        


    def grid(self,method, params = {},ver=2,griall = False): #Belirttin method için eğer belirtirsen parametlerle yoksa normal parametlerle yapıyoruz 
        
        from sklearn.model_selection import  GridSearchCV
        
        if method == 'mlp': #eğer mlp yaparsan scale ediyor datanı
            
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(self.X_train)
            X_train = scaler.transform(self.X_train)
            X_test = scaler.transform(self.X_test)
        else:
            X_train = self.X_train
            X_test = self.X_test
        
        if params:
            if method == 'rf':
                from sklearn.ensemble import RandomForestClassifier
                classifier = RandomForestClassifier()
                grid_params=params

            elif method == 'dt':
                from sklearn.tree import DecisionTreeClassifier
                classifier = DecisionTreeClassifier()
                grid_params=params

            elif method == 'mlp':
                from sklearn.neural_network import MLPClassifier
                classifier = MLPClassifier()
                grid_params=params
                
            elif method == 'lr':
                from sklearn.linear_model import LogisticRegression
                classifier = LogisticRegression()
                grid_params=params

            elif method == 'gbm':
                from sklearn.ensemble import GradientBoostingClassifier
                classifier = GradientBoostingClassifier()
                grid_params = params

            elif method == 'xgb':
                from xgboost import XGBClassifier
                classifier = XGBClassifier()
                grid_params=params

            elif method == 'lgbm':
                from lightgbm import LGBMClassifier
                classifier = LGBMClassifier()
                grid_params=params

            elif method == 'cat':
                from catboost import CatBoostClassifier
                classifier = CatBoostClassifier(silent=True)
                grid_params = params
            elif method == 'svm':
                from sklearn.svm import SVC
                classifier = SVC()
                grid_params = params
            elif method == 'knn':
                from sklearn.neighbors import KNeighborsClassifier
                classifier = KNeighborsClassifier()
                grid_params = params
            else:
                print('Unknown method')
                return
        
        else:
            if method == 'rf':
                from sklearn.ensemble import RandomForestClassifier
                classifier = RandomForestClassifier()
                grid_params = {"max_depth": [8,10,11,13,15,18],
                    "max_features": [5,10,15,20],
                     "n_estimators": [5,10,50,100,200,500],
                     "min_samples_split": [3,5,10],
                    "criterion":['entropy','gini']}

            elif method == 'dt':
                from sklearn.tree import DecisionTreeClassifier
                classifier = DecisionTreeClassifier()
                grid_params = {"max_depth": range(1,10),
                    "min_samples_split": list(range(2,50)),
                    "criterion":['gini', 'entropy']}  

            elif method == 'mlp':
                from sklearn.neural_network import MLPClassifier
                classifier = MLPClassifier()
                grid_params = {
                    'alpha':[0.1,0.01,0.001,0.005,0.0001,0.00001],
                    'hidden_layer_sizes': [(10,10,10),(45,50,60),(25,35,45),(15,15)],
                    'solver': ['lbfgs','adam','sgd'],
                    'activation': ['relu','logistic','tanh','identity']  }
                
            elif method == 'lr':
                from sklearn.linear_model import LogisticRegression
                classifier = LogisticRegression()
                grid_params = {'penalty':['l2','l1','elasticnet'],
                                'C':[0.5,0.7,0.9,1,1.1,1.2],
                                'class_weight':['None','balanced'],
                                'solver':['newton-cg','liblinear','sag']
                 }
            
            elif method == 'gbm':
                from sklearn.ensemble import GradientBoostingClassifier
                classifier = GradientBoostingClassifier()
                grid_params = {'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
                               'n_estimators': [100, 500, 1000, 1500],
                               'max_depth': [3, 5, 6],
                               'min_samples_split': [2, 5, 10, 15]}

            elif method == 'xgb':
                from xgboost import XGBClassifier
                classifier = XGBClassifier()
                grid_params = {'subsample': [0.6, 0.8, 1.0],
                               'n_estimators': [100, 500, 1000, 1500, 2000],
                               'max_depth': [3, 4, 5, 6],
                               'min_child_weight': [0.8, 0.9, 1],
                               'learning_rate': [0.1, 0.01, 0.02, 0.05]}

            elif method == 'lgbm':
                from lightgbm import LGBMClassifier
                classifier = LGBMClassifier()
                grid_params = {'subsample': [0.6, 0.8, 1.0],
                               'n_estimators': [100, 500, 1000, 1500],
                               'max_depth': [4, 5, 6],
                               'min_child_samples': [10, 20],
                               'learning_rate': [0.2, 0.1, 0.01, 0.02, 0.05],
                               'importance_type': ['gains', 'split']}

            elif method == 'cat':
                from catboost import CatBoostClassifier
                classifier = CatBoostClassifier(silent=True)
                grid_params = {'iterations': [200, 500],
                               'learning_rate': [0.01, 0.02, 0.05],
                               'depth': [3, 5, 8]}

            elif method == 'svm':
                from sklearn.svm import SVC
                classifier = SVC()
                grid_params = {'C':np.arange(1,10), 'kernel': ['linear','rbf','poly']}

            elif method == 'knn':
                from sklearn.neighbors import KNeighborsClassifier
                classifier = KNeighborsClassifier()
                grid_params = {'n_neighbors': np.arange(1, 40),
                               'metric': ['minkowski', 'euclidean', 'manhattan']}
                             
            else:
                print('Unknown method')
                return

        
        grid_cv = GridSearchCV(classifier, grid_params, cv=5, n_jobs=-1,verbose= ver)
        grid_cv_model = grid_cv.fit(X_train,self.y_train)
        
        print("En iyi parametlerler: " + str(grid_cv_model.best_params_)) #en iyi parametleri bastırıp onları kullanarak model kuruyor
        
        if method == 'rf':
        
            classifier = RandomForestClassifier(max_depth = grid_cv_model.best_params_['max_depth'],
                                                max_features = grid_cv_model.best_params_['max_features'], 
                                                n_estimators = grid_cv_model.best_params_['n_estimators'], 
                                                criterion = grid_cv_model.best_params_['criterion'])
         
        elif method == 'dt':
            
            classifier = DecisionTreeClassifier(max_depth = grid_cv_model.best_params_['max_depth'], 
                                                min_samples_split = grid_cv_model.best_params_['min_samples_split'], 
                                                criterion = grid_cv_model.best_params_['criterion'])

        elif method == 'mlp':
            classifier = MLPClassifier(alpha = grid_cv_model.best_params_['alpha'], 
                                       hidden_layer_sizes = grid_cv_model.best_params_['hidden_layer_sizes'], 
                                        solver = grid_cv_model.best_params_['solver'], 
                                       activation = grid_cv_model.best_params_['activation'])
        elif method == 'lr':
            classifier = LogisticRegression(penalty = grid_cv_model.best_params_['penalty'],
                                                C = grid_cv_model.best_params_['C'],
                                                class_weight = grid_cv_model.best_params_['class_weight'],
                                                solver = grid_cv_model.best_params_['solver']
            )
        elif method == 'gbm':
            from sklearn.ensemble import GradientBoostingClassifier
            classifier = GradientBoostingClassifier(learning_rate=grid_cv_model.best_params_['learning_rate'],
                                                    n_estimators=grid_cv_model.best_params_['n_estimators'],
                                                    max_depth=grid_cv_model.best_params_['max_depth'],
                                                    min_samples_split=grid_cv_model.best_params_['min_samples_split'])
            

        elif method == 'xgb':
            from xgboost import XGBClassifier
            classifier = XGBClassifier(
                subsample=grid_cv_model.best_params_['subsample'],
                n_estimators=grid_cv_model.best_params_['n_estimators'],
                max_depth=grid_cv_model.best_params_['max_depth'],
                min_child_weight=grid_cv_model.best_params_['min_child_weight'],
                learning_rate=grid_cv_model.best_params_['learning_rate']
                )
            

        elif method == 'lgbm':
            from lightgbm import LGBMClassifier
            classifier = LGBMClassifier(
                subsample=grid_cv_model.best_params_['subsample'],
                n_estimators=grid_cv_model.best_params_['n_estimators'],
                max_depth=grid_cv_model.best_params_['max_depth'],
                min_child_samples=grid_cv_model.best_params_[
                    'min_child_samples'],
                learning_rate=grid_cv_model.best_params_['learning_rate'],
                importance_type=grid_cv_model.best_params_['importance_type']
            )
            

        elif method == 'cat':
            from catboost import CatBoostClassifier
            classifier = CatBoostClassifier(silent=True,
                iterations=grid_cv_model.best_params_['iterations'],
                learning_rate=grid_cv_model.best_params_['learning_rate'],
                depth=grid_cv_model.best_params_['depth']
            )
        
        elif method == 'svm':
            from sklearn.svm import SVC
            classifier = SVC(C= grid_cv_model.best_params_['C'],
                            kernel=grid_cv_model.best_params_['kernel'])
        elif method == 'knn':

            from sklearn.neighbors import KNeighborsClassifier
            classifier = KNeighborsClassifier(n_neighbors=grid_cv_model.best_params_['n_neighbors'],
                                              metric=grid_cv_model.best_params_['metric'])
    
        print('Result for ',method)
        from sklearn.metrics import classification_report , accuracy_score, f1_score
        classifier.fit(X_train,self.y_train)
        y_pred = classifier.predict(X_test)
        print(classification_report(self.y_test,y_pred))
        
        if griall:
            self.tuned_acc.append(accuracy_score(self.y_test,y_pred))
            self.tuned_f1.append(f1_score(self.y_test,y_pred))
        
    def gridAll(self,clf_list=[]): #hepsinde grid uyguluyor
        self.tuned_acc = []
        self.tuned_f1 = []
        if clf_list:
            acc_columns = clf_list
        else:
            acc_columns = ['lr','dt','rf','mlp','svm','gbm','xgb','lgbm','cat','knn']
        for i in acc_columns:
            self.grid(i, ver=0, griall=True)

        self.tuned_acc = [i * 100 for i in self.tuned_acc]
        self.tuned_f1 = [ i * 100 for i in self.tuned_f1]
        zippedList = list(zip(self.tuned_acc,self.tuned_f1,self.acc,self.f1))

        accuracy = pd.DataFrame(zippedList,columns=['Tuned_Acc','Tuned_f1','Acc','F1'],
                                index= acc_columns)
        accuracy.sort_values(by="Tuned_f1").plot(kind="barh", figsize = (12,12))

class dataRegression():
    
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.splitted = False
        print("Regression class")
        
    def split(self, rs=0, shuffle = True ): #Splitting data
        from sklearn.model_selection import train_test_split
        if rs:
            X_train, X_test, y_train , y_test = train_test_split(self.x,self.y,test_size = 0.3,shuffle=shuffle, random_state=rs)
        else:
            X_train, X_test, y_train , y_test = train_test_split(self.x,self.y,shuffle=shuffle,test_size = 0.3)
            
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.splitted = True

    def selfSplit(self, X_train,X_test,y_train,y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.splitted = True

    def defaultProcesses(self): #Try some of classification models with default parameters
        
        if (self.splitted == False):
            return('Please run split before running defaultProcesses!!')

        from warnings import filterwarnings


        filterwarnings('ignore')
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor
        from sklearn.neighbors import KNeighborsRegressor

        from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
        
        acc = []
        acc_colmns  =['lr','dtc','rfc','mlpc','svm','gbm','xgb','lgbc','catboost','knn']

        lr = LinearRegression()
        lr.fit(self.X_train,self.y_train)
        y_pred = lr.predict(self.X_test)
        print('Results for default LinearRegression ')
        print(np.sqrt(mean_squared_error(self.y_test,y_pred)))
        acc.append(np.sqrt(mean_squared_error(self.y_test,y_pred)))
        
        dtc = DecisionTreeRegressor()
        dtc.fit(self.X_train,self.y_train)
        y_pred = dtc.predict(self.X_test)
        print('Results for default decision tree')
        print(np.sqrt(mean_squared_error(self.y_test,y_pred)))
        acc.append(np.sqrt(mean_squared_error(self.y_test,y_pred)))

        rfc = RandomForestRegressor()
        rfc.fit(self.X_train,self.y_train)
        y_pred = rfc.predict(self.X_test)
        print('Results for default random forest')
        print(np.sqrt(mean_squared_error(self.y_test,y_pred)))
        acc.append(np.sqrt(mean_squared_error(self.y_test, y_pred)))
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        X_train_scaled = scaler.transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
                
        mlpc = MLPRegressor()
        mlpc.fit(X_train_scaled,self.y_train)
        y_pred = mlpc.predict(X_test_scaled)
        print('Results for default MLP')
        print(np.sqrt(mean_squared_error(self.y_test,y_pred)))
        acc.append(np.sqrt(mean_squared_error(self.y_test, y_pred)))

        from sklearn.svm import SVR
        svm = SVR().fit(self.X_train, self.y_train)
        print('Results for default Gradient Boosting ')
        print(np.sqrt(mean_squared_error(self.y_test,y_pred)))
        acc.append(np.sqrt(mean_squared_error(self.y_test, y_pred)))

        gbc = GradientBoostingRegressor()
        gbc.fit(self.X_train, self.y_train)
        y_pred = gbc.predict(X_test_scaled)
        print('Results for default Gradient Boosting ')
        print(np.sqrt(mean_squared_error(self.y_test,y_pred)))
        acc.append(np.sqrt(mean_squared_error(self.y_test, y_pred)))

        xgb = XGBRegressor().fit(self.X_train, self.y_train)
        y_pred = xgb.predict(self.X_test)
        print('Results for default XGBoost ')
        print(np.sqrt(mean_squared_error(self.y_test,y_pred)))
        acc.append(np.sqrt(mean_squared_error(self.y_test, y_pred)))

        lgbm = LGBMRegressor().fit(self.X_train, self.y_train)
        print('Results for default LGBM')
        y_pred = lgbm.predict(self.X_test)
        print(np.sqrt(mean_squared_error(self.y_test,y_pred)))
        acc.append(np.sqrt(mean_squared_error(self.y_test, y_pred)))

        cat = CatBoostRegressor(silent=True).fit(self.X_train, self.y_train)

        print('Results for default CatBoost')
        y_pred = cat.predict(self.X_test)
        print(np.sqrt(mean_squared_error(self.y_test,y_pred)))
        acc.append(np.sqrt(mean_squared_error(self.y_test, y_pred)))

        knn = KNeighborsRegressor().fit(self.X_train, self.y_train)
        print('Results for default KNeighborsRegressor')
        y_pred = knn.predict(self.X_test)
        print(np.sqrt(mean_squared_error(self.y_test, y_pred)))
        acc.append(np.sqrt(mean_squared_error(self.y_test, y_pred)))

        acc = [i * 100 for i in acc]

        accuracy = pd.DataFrame({"RMSE": acc},
                                  index=acc_colmns)
        accuracy.sort_values(by="RMSE",
                       axis=0,
                       ascending=True).plot(kind="barh", color="r")
    
    # Belirttin method için eğer belirtirsen parametlerle yoksa normal parametlerle yapıyoruz
    def grid(self, method, params={}, ver=2, griall=False):

        from sklearn.model_selection import GridSearchCV

        if method == 'mlp':  # eğer mlp yaparsan scale ediyor datanı

            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(self.X_train)
            X_train = scaler.transform(self.X_train)
            X_test = scaler.transform(self.X_test)
        else:
            X_train = self.X_train
            X_test = self.X_test

        if params:
            if method == 'rf':
                from sklearn.ensemble import RandomForestRegressor
                classifier = RandomForestRegressor()
                grid_params = params

            elif method == 'dt':
                from sklearn.tree import DecisionTreeRegressor

                classifier = DecisionTreeRegressor()
                grid_params = params

            elif method == 'mlp':
                from sklearn.neural_network import MLPRegressor
                classifier = MLPRegressor()
                grid_params = params

            elif method == 'lr':
                from sklearn.linear_model import LinearRegression
                classifier = LinearRegression()
                grid_params = params

            elif method == 'gbm':
                from sklearn.ensemble import GradientBoostingRegressor
                classifier = GradientBoostingRegressor()
                grid_params = params

            elif method == 'xgb':
                from xgboost import XGBRegressor
                classifier = XGBRegressor()
                grid_params = params

            elif method == 'lgbm':
                from lightgbm import LGBMRegressor
                classifier = LGBMRegressor()
                grid_params = params

            elif method == 'cat':
                from catboost import CatBoostRegressor
                classifier = CatBoostRegressor(silent=True)
                grid_params = params
            elif method == 'svm':
                from sklearn.svm import SVR
                classifier = SVR()
                grid_params = params
            elif method == 'knn':
                from sklearn.neighbors import KNeighborsRegressor
                knn = KNeighborsRegressor()
                grid_params = params
            
            else:
                print('Unknown method')
                return

        else:
            if method == 'rf':
                from sklearn.ensemble import RandomForestRegressor
                classifier = RandomForestRegressor()
                grid_params = {"max_depth": [8, 10, 11, 13, 15, 18],
                               "max_features": [5, 10, 15, 20],
                               "n_estimators": [5, 10, 50, 100, 200, 500],
                               "min_samples_split": [3, 5, 10],
                               "criterion": ['mse', 'mae']}

            elif method == 'dt':
                from sklearn.tree import DecisionTreeRegressor
                classifier = DecisionTreeRegressor()
                grid_params = {"max_depth": range(1, 10),
                               "min_samples_split": list(range(2, 10)),
                               "criterion": ['mse', 'mae']}

            elif method == 'mlp':
                from sklearn.neural_network import MLPRegressor
                classifier = MLPRegressor()
                grid_params = {
                    'alpha': [0.1, 0.01, 0.001, 0.005, 0.0001, 0.00001],
                    'hidden_layer_sizes': [(10, 10, 10), (45, 50, 60), (25, 35, 45), (15, 15),(100,),(100,100)],
                    'solver': ['lbfgs', 'adam', 'sgd'],
                    'activation': ['relu', 'logistic', 'tanh', 'identity']}

            elif method == 'gbm':
                from sklearn.ensemble import GradientBoostingRegressor
                classifier = GradientBoostingRegressor()
                grid_params = {'loss' : ['ls','lad','huber','quantile'],
                            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
                            'n_estimators': [100, 500, 1000, 1500],
                            'max_depth': [3, 5, 6],
                            'min_samples_split': [2, 5, 10, 15],
                            'subsample':[0.6,1.0]}


            elif method == 'xgb':
                from xgboost import XGBRegressor
                classifier = XGBRegressor()
                grid_params = {'colsample_bytree':[0.6,1.0],
                        'n_estimators':[100,200,500,1000],
                        'max_depth':[4,5,6, 7],
                        'min_child_weight':[0.8,0.9,1],
                        'learning_rate': [0.1,0.01,0.02,0.05]}

            elif method == 'lgbm':
                from lightgbm import LGBMRegressor
                classifier = LGBMRegressor()
                grid_params = {'subsample': [0.6, 0.8, 1.0],
                               'n_estimators': [100, 500, 1000, 1500],
                               'max_depth': [4, 5, 6, 7],
                               'min_child_samples': [10, 20],
                               'learning_rate': [0.2, 0.1, 0.01, 0.02, 0.05],
                               'importance_type': ['gains', 'split']}

            elif method == 'cat':
                from catboost import CatBoostRegressor
                classifier = CatBoostRegressor(silent=True)
                grid_params = {'iterations': [200, 500],
                               'learning_rate': [0.01, 0.02, 0.05],
                               'depth': [3, 5, 8]}

            elif method == 'svm':
                from sklearn.svm import SVR
                classifier = SVR()
                grid_params = {'C':np.arange(0.1,2,0.1), 'kernel': [
                    'linear', 'rbf', 'poly']}

            elif method == 'knn':
                from sklearn.neighbors import KNeighborsRegressor
                classifier = KNeighborsRegressor()
                grid_params = {'n_neighbors': np.arange(1,40),
              'weights':['uniform','distance'],
             'metric':['minkowski','euclidean','manhattan']}


            else:
                print('Unknown method')
                return

        grid_cv = GridSearchCV(classifier, grid_params,
                               cv=5, n_jobs=-1, verbose=ver)
        grid_cv_model = grid_cv.fit(X_train, self.y_train)

        # en iyi parametleri bastırıp onları kullanarak model kuruyor
        print("En iyi parametlerler: " + str(grid_cv_model.best_params_))

        if method == 'rf':

            classifier = RandomForestRegressor(max_depth=grid_cv_model.best_params_['max_depth'],
                                                max_features=grid_cv_model.best_params_[
                                                    'max_features'],
                                                n_estimators=grid_cv_model.best_params_[
                                                    'n_estimators'],
                                               min_samples_split=grid_cv_model.best_params_[
                                                   'min_samples_split'],
                                                criterion=grid_cv_model.best_params_['criterion'])

        elif method == 'dt':

            classifier = DecisionTreeRegressor(max_depth=grid_cv_model.best_params_['max_depth'],
                                                min_samples_split=grid_cv_model.best_params_[
                                                    'min_samples_split'],
                                                criterion=grid_cv_model.best_params_['criterion'])

        elif method == 'mlp':
            classifier = MLPRegressor(alpha=grid_cv_model.best_params_['alpha'],
                                       hidden_layer_sizes=grid_cv_model.best_params_[
                                           'hidden_layer_sizes'],
                                       solver=grid_cv_model.best_params_[
                                           'solver'],
                                       activation=grid_cv_model.best_params_['activation'])
        
        elif method == 'gbm':
            from sklearn.ensemble import GradientBoostingRegressor
            classifier = GradientBoostingRegressor(learning_rate=grid_cv_model.best_params_['learning_rate'],
                                                    n_estimators=grid_cv_model.best_params_[
                                                        'n_estimators'],
                                                    max_depth=grid_cv_model.best_params_[
                                                        'max_depth'],
                                                    min_samples_split=grid_cv_model.best_params_['min_samples_split'],
                                                    loss=grid_cv_model.best_params_['loss'],
                                                   subsample= grid_cv_model.best_params_['subsample'])

        elif method == 'xgb':
            from xgboost import XGBRegressor
            classifier = XGBRegressor(
                colsample_bytree=grid_cv_model.best_params_[
                    'colsample_bytree'],
                n_estimators=grid_cv_model.best_params_['n_estimators'],
                max_depth=grid_cv_model.best_params_['max_depth'],
                min_child_weight=grid_cv_model.best_params_[
                    'min_child_weight'],
                learning_rate=grid_cv_model.best_params_['learning_rate']
            )

        elif method == 'lgbm':
            from lightgbm import LGBMRegressor
            classifier = LGBMRegressor(
                subsample=grid_cv_model.best_params_['subsample'],
                n_estimators=grid_cv_model.best_params_['n_estimators'],
                max_depth=grid_cv_model.best_params_['max_depth'],
                min_child_samples=grid_cv_model.best_params_[
                    'min_child_samples'],
                learning_rate=grid_cv_model.best_params_['learning_rate'],
                importance_type=grid_cv_model.best_params_['importance_type']
            )

        elif method == 'cat':
            from catboost import CatBoostRegressor
            classifier = CatBoostRegressor(silent=True,
                                            iterations=grid_cv_model.best_params_[
                                                'iterations'],
                                            learning_rate=grid_cv_model.best_params_[
                                                'learning_rate'],
                                            depth=grid_cv_model.best_params_[
                                                'depth']
                                            )

        elif method == 'svm':
            from sklearn.svm import SVR
            classifier = SVR(C=grid_cv_model.best_params_['C'],
                             kernel=grid_cv_model.best_params_['kernel'])
        elif method == 'knn':
            from sklearn.neighbors import KNeighborsRegressor
            knn = KNeighborsRegressor(
                n_neighbors=grid_cv_model.best_params_['n_neighbors'],
                weights=grid_cv_model.best_params_['weights'],
                metric=grid_cv_model.best_params_['metric'])
 

        print('Result for ', method)
        from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
        classifier.fit(X_train, self.y_train)
        y_pred = classifier.predict(X_test)

        print(np.sqrt(mean_squared_error(self.y_test, y_pred)))
        if griall:
            self.rmse.append(np.sqrt(mean_squared_error(self.y_test, y_pred)))

    def gridAll(self,clf_list=[]): #hepsinde grid uyguluyor
        self.rmse = []
        if clf_list:
            acc_columns = clf_list
        else:
            acc_columns = ['dt','rf','svm','gbm','xgb','lgbm','cat','knn']
        for i in acc_columns:
            self.grid(i, ver=0, griall=True)

        self.rmse = [i * 100 for i in self.rmse]
        
        accuracy = pd.DataFrame({"Acc": self.rmse},
                                index=acc_columns)
        accuracy.sort_values(by="Acc",
                               axis=0,
                               ascending=True).plot(kind="barh", color="r")
