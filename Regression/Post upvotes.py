import numpy as np
import pandas as pd
import os
import seaborn as sns
os.chdir("D:/Documents/Education/Data Science ProDegree Imarticus/Python projects/Analytics Vidya and Kaggle/predict upvotes")
pd.set_option('display.float_format', lambda x: '%.5f' % x)

impraw=pd.read_csv("train.csv")
imptest=pd.read_csv("test.csv")
impraw["Source"]="Train"
imptest["Source"]="Test"
imptest["Upvotes"] = 0
ids=imptest["ID"]

fullraw=pd.concat([impraw, imptest], axis=0)
##outlier detection -----------------------------------------------------------------------------
fullraw.hist("Answers")
fullraw["Answers_log"]=np.where(fullraw["Answers"]==0,0,np.log(fullraw["Answers"]))
fullraw.hist("Answers_log")

fullraw.hist("Views")
fullraw["Views_log"]=np.where(fullraw["Views"]==0,0,np.log(fullraw["Views"]))
fullraw.hist("Views_log")

fullraw.hist("Reputation")
fullraw["Reputation_log"]=np.where(fullraw["Reputation"]==0,0,np.log(fullraw["Reputation"]))
fullraw.hist("Reputation_log")

fullraw.hist("Upvotes")
fullraw["Upvotes_log"]=np.where(fullraw["Upvotes"]==0,0,np.log(fullraw["Upvotes"]))
fullraw.hist("Upvotes_log")

fullraw.drop(["Username", "ID", "Reputation","Upvotes", "Views", "Answers"], axis=1, inplace=True)
# ------------------------------------------------------------------------------------------------

fullraw.isnull().sum()

fullraw2=pd.get_dummies(fullraw, drop_first=True)

TrainAll=fullraw2[fullraw2["Source_Train"]==1].drop(["Source_Train"], axis=1).copy()
FinalTest=fullraw2[fullraw2["Source_Train"]==0].drop(["Source_Train"], axis=1).copy()


featuresnames=[i for i in TrainAll.columns if(i!="Upvotes_log")]
depdnt=TrainAll["Upvotes_log"]
from sklearn.model_selection import train_test_split
TrainX, TestX, TrainY, TestY= train_test_split(TrainAll[featuresnames], depdnt, train_size=0.70, random_state=123)


#from sklearn.ensemble import RandomForestRegressor
#m1=RandomForestRegressor(random_state=4, n_estimators=700, min_samples_leaf=3, max_features=7,min_samples_split=15, n_jobs=-1, warm_start=True).fit(TrainX, TrainY)
#Testpred=m1.predict(TestX)
#Testpred=np.exp(Testpred)
#RFRMSE=np.sqrt(np.mean((TestY-Testpred)**2))

#{'max_features': 4, 'min_samples_leaf': 5, 'min_samples_split': 15, 'n_estimators': 600}
#{'max_features': 4, 'min_samples_leaf': 3, 'min_samples_split': 15, 'n_estimators': 700}

#variable importance
#varimp=pd.concat([pd.DataFrame(m1.feature_importances_), pd.DataFrame(TrainX.columns)], axis=1)
#varimp.columns=["values", "columnsnames"]
#varimp.sort_values(["values"], inplace=True)
#sns.scatterplot(x=varimp["values"], y=varimp["columnsnames"])


#stacking regressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor


estimator=[('RF',RandomForestRegressor(random_state=4, n_estimators=700, min_samples_leaf=3, max_features=7,min_samples_split=15, warm_start=True)), ('KNN',KNeighborsRegressor(n_neighbors=7))]
ms=StackingRegressor(estimators=estimator, final_estimator=LinearRegression()).fit(TrainX, TrainY)
Testpred=ms.predict(TestX)
Testpred=np.exp(Testpred)
STRMSE=np.sqrt(np.mean((TestY-Testpred)**2))

#out
FinalTest.drop(["Upvotes_log"], axis=1, inplace= True)
finalpred=ms.predict(FinalTest)
finalpred=np.exp(finalpred)
submission=pd.DataFrame({"ID":ids, "Upvotes":finalpred})
submission.to_csv("uppy03logtrGCVST.csv", index=False)

#grid search
#GridX,DX, GridY, DY=train_test_split(TrainAll[featuresnames], depdnt, train_size=0.10, random_state=4)
#
#from sklearn.model_selection import GridSearchCV
#parameters={"n_estimators":range(100,800,100), "min_samples_leaf":range(1,20,2), "min_samples_split":range(5,20,5), "max_features":range(1,5,1)}
#Grid=GridSearchCV(estimator=RandomForestRegressor(random_state=4), param_grid=parameters,scoring='r2', cv=3,n_jobs=-2, verbose=3, pre_dispatch="2*n_jobs").fit(GridX, GridY)
#Griddf=pd.DataFrame.from_dict(Grid.cv_results_)
#Griddf.to_csv("Griddf.csv", index=False)
#Griddf=pd.read_csv("Griddf.csv") #{'min_samples_leaf': 7, 'n_estimators': 700} 
