import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

os.chdir("D:/Docs/Education/Data Science ProDegree Imarticus/Python projects/Analytics Vidya and Kaggle/titanic")

impraw=pd.read_csv("train.csv")
imptest=pd.read_csv("test.csv")
#data preparation_______________________________________________________________________________________############


impraw["Source"]="Train"
imptest["Source"]="Test"
imptest["Survived"]=0
fulldata=pd.concat([impraw, imptest], axis=0)
id=imptest["PassengerId"]

#missing values check
fulldata.isnull().sum()

#imputing missing values
onlytrain=fulldata["Source"]=="Train"
halfrows=0.5*fulldata[onlytrain].shape[0]
for i in fulldata.columns:
    totalNAs=fulldata.loc[onlytrain,i].isnull().sum()
    if(totalNAs<halfrows):
        if(fulldata[i].dtype=="object"):
            tempmode=fulldata.loc[onlytrain, i].mode()[0]
            print(i, " is a category variable using mode to impute, mode value: ", tempmode, "total NA: ", totalNAs)
            fulldata[i].fillna(tempmode, inplace=True)
        else:
            tempmedian=fulldata.loc[onlytrain, i].median()
            print(i, " is not a category variable using median to impute, median value: ", tempmedian, "total NA: ", totalNAs)
            fulldata[i].fillna(tempmedian, inplace=True)
    else:
        if(i!=["Source", "Survived"]):
            fulldata.drop([i], axis=1, inplace=True)
            
            
fulldata.isnull().sum() #rechecking
fulldata.drop([ "Ticket", "PassengerId"], axis=1, inplace=True)
fulldata["Totalpass"]=fulldata["Parch"]+fulldata["SibSp"]+1

#passenger stats
fulldata["Age"].describe()
fulldata[["Sex","Age","Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Age", ascending=True)

condpb=[(fulldata["Age"]<16)|(fulldata["Age"]>64),\
        (fulldata["Age"]<25)&(fulldata["Sex"]=="female"),\
        (fulldata["Age"]<30)&(fulldata["Sex"]=="male")&((fulldata["Pclass"]==1)|(fulldata["Pclass"]==2)),\
        (fulldata["Age"]>16)&(fulldata["Sex"]=="male")]
choicepb=[1,2,3,4]   
fulldata["PassengerSurvChance"]=np.select(condpb, choicepb)
fulldata.loc[fulldata["PassengerSurvChance"]==0, "PassengerSurvChance"]=5

#age bands
#creating age bands
fulldata["AgeBands"]=pd.cut(fulldata["Age"], 5)
fulldata[["AgeBands", "Survived"]].groupby(["AgeBands"], as_index=False).mean().sort_values(by="AgeBands", ascending=True) 

condage=[fulldata["Age"]<=16, (fulldata["Age"]>16)&(fulldata["Age"]<=32),(fulldata["Age"]>32)&(fulldata["Age"]<=48),\
                              (fulldata["Age"]>48)&(fulldata["Age"]<=64),fulldata["Age"]>64]
choiceage=[1,2,3,4,5]
fulldata["Age"]=np.select(condage, choiceage)
#gender encoding
fulldata.loc[fulldata["Sex"]=="male", "Sex"]=1
fulldata.loc[fulldata["Sex"]=="female", "Sex"]=2

#Passanger class
fulldata[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Pclass",ascending=True)  #holding this variable

#copassengers
fulldata["totalpassbins"]=pd.cut(fulldata["Totalpass"], 5)
fulldata[["totalpassbins","Survived"]].groupby(["totalpassbins"], as_index=False).mean().sort_values(by="Survived", ascending=False)

condco=[(fulldata["Totalpass"]<=3),\
        (fulldata["Totalpass"]>3)&(fulldata["Totalpass"]<=5),\
        (fulldata["Totalpass"]>5)]
choiceco=[1,2,3]   
fulldata["Copass"]=np.select(condco, choiceco)
fulldata.drop(["AgeBands", "Totalpass","totalpassbins"], axis=1, inplace=True)

# #name logics
# fulldata["Title"]=fulldata.Name.str.extract(pat='([A-Za-z]+)\.', expand=False) 

# pd.crosstab(fulldata["Title"], fulldata["Sex"])
# fulldata["Title"].replace(["Countess","Jonkheer", "Mlle", "Sir", "Mme", "Don", "Dona", "Lady",], "Rare", inplace=True)
# fulldata["Title"].replace(["Miss","Mrs","Ms", ], "Female", inplace=True)
# fulldata["Title"].replace(["Master","Mr"], "Male", inplace=True)
# pd.crosstab(fulldata["Title"], fulldata["Survived"]).sort_values(by=0, ascending=False) #0=death

# condname=[fulldata["Title"]=="Female",fulldata["Title"]=="Rev", fulldata["Title"]=="Rare", fulldata["Title"]=="Dr", fulldata["Title"]=="Male"]
# choicename=[1,2,3,4,5]   
# fulldata["People"]=np.select(condname, choicename)
# fulldata.loc[fulldata["People"]==0, "People"]=6
#embarkments
fulldata[["Embarked", "Survived"]].groupby(["Embarked"], as_index=False).mean()

condembk=[fulldata["Embarked"]=="C",fulldata["Embarked"]=="Q", fulldata["Embarked"]=="S"]
choiceembk=[1,2,3]   
fulldata["PortEmbarked"]=np.select(condembk, choiceembk)
fulldata.drop(["Name", "Embarked"], axis=1, inplace=True)
#############_______________________________________________________________________________________############


#observations
#1. Fare does not have much impact with Survival
#2. But Pclass has impact with Survival (which means fare is getting impacted from number of passengers per ticket)

#correlation check

cordf=fulldata.corr()
sns.heatmap(data=cordf, xticklabels=cordf.columns, yticklabels=cordf.columns, cmap="BrBG")



Trainraw=fulldata[fulldata["Source"]=="Train"].drop(["Source"], axis=1).copy()
Test=fulldata[fulldata["Source"]=="Test"].drop(["Source"], axis=1).copy()

#sampling train and test from train and dependent and independent variables
independentvar=[x for x in Trainraw.columns if (x!="Survived")]
dependentvar=Trainraw["Survived"]


TrainX, ValidX, TrainY, ValidY=train_test_split(Trainraw[independentvar], dependentvar, train_size=0.80, random_state=123)

TestX=Test.drop(["Survived"], axis=1).copy()
TestY=Test["Survived"]

#_____________________________________________________________________________________________________________________________


# RBS=RobustScaler().fit(TrainX)
# TrainX_scaled=RBS.transform(TrainX)
# ValidX_Scaled=RBS.transform(ValidX)
# TestX_scaled=RBS.transform(TestX)

#modelbuilding#_______________________________________________________________________________________________________________

model=RandomForestClassifier(random_state=123, n_jobs=-2, max_features=4, min_samples_leaf=3,min_samples_split=5, n_estimators=300).fit(TrainX_scaled,TrainY)
Validpred_prob=model.predict_proba(ValidX_Scaled)[:,1]
Validpred=np.where(Validpred_prob>0.7,1,0)
#{'max_features': 8, 'min_samples_leaf': 3, 'min_samples_split': 6, 'n_estimators': 800}
#{'max_features': 4, 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 500}
#{'max_features': 4, 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 300}



confusion_matrix(Validpred, ValidY)
f1_score(Validpred, ValidY)
RFS=sum(np.diagonal(confusion_matrix(Validpred, ValidY)))/ValidY.shape[0]*100
#_______________________________________________________________________________

#variable importance

# varimp=pd.concat([pd.DataFrame(model.feature_importances_),pd.DataFrame(TrainX.columns)],axis=1)
# varimp.columns=["values", "features"]
# varimp.sort_values(by="values", ascending=True, inplace=True)
# sns.scatterplot(x=varimp["values"],y=varimp["features"])


#_____________________________________________________________________________________________________________________________
#Stacking Classifier



est=estimators = [('rf', RandomForestClassifier(random_state=4, max_features="auto", min_samples_leaf=5,min_samples_split=4, n_estimators=1500)), 
                  ('lr', LogisticRegression(penalty="l2", max_iter=500))]
meta=GradientBoostingClassifier(learning_rate=0.01, n_estimators=500, max_features="sqrt", min_samples_leaf=5)
ms=StackingClassifier(estimators=est, final_estimator=meta, stack_method="predict_proba", passthrough=True).fit(TrainX, TrainY)


Validpred_prob=ms.predict_proba(ValidX)[:,1]
Validpred=np.where(Validpred_prob>0.50,1,0)
confusion_matrix(Validpred, ValidY)
f1_score(Validpred, ValidY)
RFS=sum(np.diagonal(confusion_matrix(Validpred, ValidY)))/ValidY.shape[0]*100
#_______________________________________________________________________________


#_____________________________________________________________________________________________________________________________

#finaloutput
Test_pred_prob=ms.predict_proba(TestX)[:,1]
Survived=np.where(Test_pred_prob>0.50,1,0)
submission=pd.DataFrame({"PassengerId":id, "Survived":Survived})
submission.to_csv("titanic01_RFLRGB_GB.csv", index=False)


#GRID search RF__________________________________________________________________________________________________________________

parameters={"n_estimators":range(500,2000,200), "min_samples_leaf":range(1,7,2), "min_samples_split":range(1,15,3)}
Grid=GridSearchCV(estimator=RandomForestClassifier(random_state=4), param_grid=parameters, n_jobs=-2, scoring="accuracy", verbose=5, cv=3, pre_dispatch=10).fit(TrainX, TrainY)
Griddf_rf=pd.DataFrame.from_dict(Grid.cv_results_)
Griddf_rf.to_csv("Griddf3.csv")

#GRID search RF__________________________________________________________________________________________________________________

parameters={"learning_rate":[0.1,0.01,0.001,1.0], "min_samples_leaf":range(1,7,2), 
            "n_estimators":range(100,2000,200), "max_features":["auto", "sqrt", "log2"]}
Grid=GridSearchCV(estimator=GradientBoostingClassifier(random_state=4), param_grid=parameters, n_jobs=-2, scoring="accuracy", verbose=5, cv=5,).fit(TrainX, TrainY)
Griddf_gb=pd.DataFrame.from_dict(Grid.cv_results_)
Griddf_gb.to_csv("Griddf3.csv")

