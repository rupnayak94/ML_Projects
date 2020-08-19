import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle


#reading data
pre_train=pd.read_csv("train.csv")
pre_test=pd.read_csv(("test.csv"))

pre_train.info()
pre_train["Source"]="Train"
pre_test["Source"]="Test"
ids=pre_test["PassengerId"]
fulldata=pd.concat([pre_train, pre_test], axis=0, sort=True).reset_index(drop=True)

fulldata.isnull().sum()


#-----------------------------
#missing value removal for AGE
#-----------------------------

#imputing missing values
fulldata.corr()["Age"].abs().sort_values(ascending=False)
#age as correlation with Pclass
#mediam group age and pclass

fulldata.groupby(['Sex', 'Pclass']).median()['Age']
#imputing age by median of pclass and sex groups
fulldata["Age"]=fulldata.groupby(['Sex', 'Pclass'])["Age"].apply(lambda x: x.fillna(x.median()))



#-------------------------------
#missing value removal for CABIN
#-------------------------------

fulldata.corr()["Pclass"].abs().sort_values(ascending=False)
fulldata["Deck"]=fulldata["Cabin"].apply(lambda s: s[0] if pd.notnull(s) else "M")
cab_surv=pd.DataFrame(fulldata.groupby(["Deck", "Pclass"]).count()["Survived"]).reset_index()
deck_tot=pd.DataFrame(fulldata.groupby(["Pclass","Deck"]).count()["Ticket"]).reset_index()
pclass_tot=pd.DataFrame(fulldata.groupby(["Pclass"]).count()["Ticket"]).reset_index()
fulldata.groupby(["Pclass"]).count()["Ticket"]

#percentage of passenger per deck per class
temp_per=[]
for i in range(1,4):
    print(i)
    temp=deck_tot.loc[deck_tot["Pclass"]==i, "Ticket"]/int(pclass_tot.loc[pclass_tot["Pclass"]==i, "Ticket"])*100
    print(temp)
    temp_per.extend(temp)
deck_tot["per"]=temp_per

#pclass and deck combination
deck_tot["cab"]=deck_tot["Pclass"].astype(str)+deck_tot["Deck"] 

%matplotlib inline
#ploting passenger density per Pclass per Deck
plt.figure()
plt.bar(deck_tot["cab"],deck_tot["per"] ,  color='peru')
plt.xticks(deck_tot["cab"])

#ploting total passenger per class 
plt.figure()
plt.bar(deck_tot["Pclass"],deck_tot["per"] ,  color='lightseagreen')
plt.xticks(deck_tot["Pclass"]) 
plt.figure()

#ploting total Survived passenger per class 
plt.bar(cab_surv["Pclass"], cab_surv["Survived"], color="indigo")
plt.xticks(cab_surv["Pclass"]) 
plt.show()

#missing value assignment
fulldata["cab"]=fulldata["Pclass"].astype(str)+fulldata["Deck"] 
idx = fulldata[fulldata['cab'] == '1T'].index
fulldata.loc[idx, 'cab'] = '1A'

idx = fulldata[fulldata['cab'] == '1M'].index
fulldata.loc[idx, 'cab'] = '1B'          #as below the promenade deck

idx = fulldata[fulldata['cab'] == '2M'].index
fulldata.loc[idx, 'cab'] = '2E'          #as 2E has the minimum passengers

idx = fulldata[fulldata['cab'] == '3M'].index
fulldata.loc[idx, 'cab'] = '3G'          #above the Orlop deck

fulldata.drop(["Cabin","Deck"], axis=1, inplace=True)

#-------------------------------
#missing value removal for Fare
#-------------------------------

fulldata.groupby(["Pclass", "cab"])["Fare"].median()
fulldata["Fare"]=fulldata.groupby(["Pclass", "cab"])["Fare"].apply(lambda x: x.fillna(x.median()))

#-------------------------------
#missing value removal for Embarked
#-------------------------------
mode=pre_train["Embarked"].mode()
fulldata["Embarked"].fillna(mode, inplace=True)


#-------------------------------
#Total passengers per ticket
#-------------------------------
fulldata["T_pass_pr_tckt"]=fulldata["SibSp"]+fulldata["Parch"]+1
fulldata.drop(["SibSp", "Parch"], axis=1, inplace=True)
#-------------------------------
#Title
#-------------------------------
fulldata["Title"]=fulldata["Name"].str.split(', ', expand=True)[1].str.split(".", expand=True)[0]
fulldata["Title"].value_counts()
fulldata.groupby(["Title"])["Survived"].count()

fulldata["Title"]=fulldata["Title"].replace(["Miss", "Mrs", "Ms", "Mme", "Mlle", "Lady", "the Countess", "Dona"], "Miss/Mrs")
fulldata["Title"]=fulldata["Title"].replace(["Capt", "Col", "Dr", "Major", "Don", "Jonkheer", "Rev", "Sir"], "Officers")

fulldata.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)
#-------------------------------
#Bucketting of values
#-------------------------------




#-------------------------------
#Dummies
#-------------------------------
fulldata2=pd.get_dummies(fulldata, drop_first=True)


#-------------------------------
#Train Test and Valid Splits
#-------------------------------
Train=fulldata2[fulldata2["Source_Train"]==1].drop(["Source_Train"], axis=1).copy()
Test=fulldata2[fulldata2["Source_Train"]==0].drop(["Source_Train"], axis=1).copy()

indfeat=[x for x in Train.columns if x!="Survived"]
depvar=Train["Survived"]
TrainX, ValidX, TrainY, ValidY=train_test_split(Train[indfeat], depvar, train_size=0.7,
                                                random_state=4)

TestX=Test.drop(["Survived"], axis=1).copy()
TestY=Test["Survived"].copy()

#-------------------------------
#Scaling of data
#-------------------------------
StdSclr=StandardScaler().fit(TrainX)
TrainX_Std=StdSclr.transform(TrainX)
ValidX_Std=StdSclr.transform(ValidX)
TestX_Std=StdSclr.transform(TestX)

TrainX_Std=pd.DataFrame(TrainX_Std, columns=TrainX.columns)
ValidX_Std=pd.DataFrame(ValidX_Std, columns=ValidX.columns)
TestX_Std=pd.DataFrame(TestX_Std, columns=TestX.columns)

#-------------------------------
#Default multi-modelling
#-------------------------------

logi=LogisticRegression(penalty="elasticnet",l1_ratio=0.5,solver="saga", random_state=4, n_jobs=-1)
rf=RandomForestClassifier(random_state=4, n_jobs=-1, max_features="auto")
gb=GradientBoostingClassifier(random_state=4, max_features="auto")
svc=SVC(random_state=4, kernel='rbf')
ex=ExtraTreesClassifier(random_state=4, n_jobs=-1, max_features="auto")



stac_class_estimtr=[("logi", LogisticRegression()), ("rf", RandomForestClassifier( random_state=4, n_jobs=-1)),
                    ("gb",GradientBoostingClassifier(random_state=4)), 
                     ("svc", SVC(kernel="rbf", random_state=4)),
                     ("ex", ExtraTreesClassifier(random_state=4, n_jobs=-1))]
Stc=StackingClassifier(estimators=stac_class_estimtr, final_estimator=GradientBoostingClassifier(random_state=4))

models_dict={"Logistic_Regression":logi, "Random_Forest": rf,
             "Gradient_Boosting": gb,"Support_Vector":svc, 
             "Extra_trees":ex,"Stacking_Classifier": Stc}
                       
model_perf=pd.DataFrame()
for clf_name, clf in zip(models_dict.keys(), models_dict.values()):
    clf.fit(TrainX_Std, TrainY)
    valid_pred=clf.predict(ValidX_Std)
    perf=pd.Series({"model":clf_name, "acc_scr":accuracy_score(ValidY, valid_pred)})
    model_perf=model_perf.append(perf, ignore_index=True)

#-------------------------------
#Grid Search and Hyper-parameter tunning
#-------------------------------

#Grid01

ensemble_clf=[rf, ex, gb, svc] 
params1={"max_depth": range(5,30,5), "min_samples_leaf": range(1,30,2),
         "n_estimators":range(100,2000,200)}
params2={"criterion":["gini", "entropy"],"max_depth": range(5,30,5), 
         "min_samples_leaf": range(1,30,2), "n_estimators":range(100,2000,200)}
params3={"learning_rate":[0.001,0.01,0.1], "n_estimators":range(1000,3000,200)}
params4={"kernel":["rbf", "poly"], "gamma": ["auto", "scale"], "degree":range(1,6,1)}

parameters_list=[params1, params2, params3, params4]
model_log=["_rf", "_ex", "_gb", "_svc"]

for i in range(len(ensemble_clf)):
    Grid=GridSearchCV(estimator=ensemble_clf[i], param_grid=parameters_list[i], 
                      n_jobs=-1, cv=3, verbose=3).fit(TrainX_Std, TrainY)
    globals()['Grid%s' % model_log[i]]=pd.DataFrame(Grid.cv_results_)   

Grid_ex.to_pickle("Grid_Extratrees.pkl")
Grid_rf.to_pickle("Grid_RandomForest.pkl")
Grid_svc.to_pickle("Grid_SVC.pkl")
Grid_gb.to_pickle("Grid_GB.pkl") 
    
#Grid02
params5={"learning_rate":[0.001,0.0001,0.01], "n_estimators":range(1000,3000,200),
         "subsample":[0.5,0.8]}
Gridgb=GridSearchCV(estimator=gb, param_grid=params5, 
                      n_jobs=-1, cv=3, verbose=3).fit(TrainX_Std, TrainY)
Grid_gb2=pd.DataFrame(Gridgb.cv_results_)

Grid_gb2.to_pickle("./Grid_GB2.pkl") 


# ###read for pickles
# Grid_ex=pd.read_pickle("Grid_Extratrees.pkl")
# Grid_rf=pd.read_pickle("Grid_RandomForest.pkl")
# Grid_sv=pd.read_pickle("Grid_SVC.pkl")
# Grid_gb=pd.read_pickle("Grid_GB2.pkl")

#training the final estimator



stac_class_estimtr=[("logi", LogisticRegression(penalty="elasticnet",l1_ratio=0.5,solver="saga",
                                                random_state=4, n_jobs=-1)), 
                    ("rf", RandomForestClassifier(n_estimators=500, min_samples_leaf=3, 
                                                  max_depth=5, max_features="auto",
                                                  random_state=4, n_jobs=-1)),
                    ("gb",GradientBoostingClassifier(learning_rate=0.001, n_estimators=1600,
                                                    subsample=0.8, random_state=4)), 
                     ("svc", SVC(kernel="rbf", random_state=4, gamma="scale", probability=True)),
                     ("ex", ExtraTreesClassifier(n_estimators=700, min_samples_leaf=3, 
                                                  max_depth=25, max_features="auto",
                                                  random_state=4, n_jobs=-1))]
Stc2=StackingClassifier(estimators=stac_class_estimtr, 
                       final_estimator=GradientBoostingClassifier(learning_rate=0.001, 
                                 n_estimators=2400, subsample=0.8, random_state=4),
                                stack_method="predict_proba", verbose=3)
Stc_m=Stc2.fit(TrainX_Std, TrainY)
valid_pred_pro=Stc_m.predict_proba(ValidX_Std)[:,1]
valid_pred=np.where(valid_pred_pro>=0.36,1,0)
metrics={"acc": accuracy_score(ValidY, valid_pred),"f1": f1_score(ValidY, valid_pred),
         "confusion_mat": confusion_matrix(ValidY, valid_pred)}
metrics
fpr, tpr, thresholds=roc_curve(ValidY, valid_pred_pro)

#ruc curve
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

#finding the best cutoff:
roc_dict={"tpr": tpr, "fpr": fpr,"cutoff":thresholds, "maxdiff": tpr-fpr,
          "min_dist": np.sqrt(((1-tpr)**2)+((0-fpr)**2))} 
roc_df=pd.DataFrame(roc_dict)


#rejoining
full_Train_StdX=pd.concat([TrainX_Std, ValidX_Std], axis=0)
full_Train_StdY=pd.concat([TrainY, ValidY], axis=0)

Stc_fm=Stc2.fit(full_Train_StdX,full_Train_StdY)

#final submission
test_pred_prob=Stc_fm.predict_proba(TestX_Std)[:,1]
test_pred=np.where(test_pred_prob>=0.75,1,0)
submission=pd.DataFrame({"PassengerId": ids, "Survived":test_pred})
submission.to_csv("titanic2_stck01.csv", index=False)

file02=open("stackingC", "wb")
pickle.dump(Stc2, file02)
file02.close()
