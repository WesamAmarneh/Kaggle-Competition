#import needed library
import numpy as np
import pandas as pd
import matplotlib as plt

#get the datatrain & datatest file
datatrain = pd.read_csv("../input/train.csv")
datatest = pd.read_csv("../input/test.csv")


def impute_v2a1(coulmns):
    v2a1 = coulmns[0]
    rooms = coulmns[1]
    paredpreb = coulmns[2]
    pisonotiene = coulmns[3]
    if pd.isna(v2a1):
        if rooms == 1:
            if paredpreb == 0:
                if pisonotiene == 0:
                    return 76667
                elif pisonotiene == 1:
                    return 15000
            elif paredpreb == 1:
                return 55378
        elif rooms == 2:
            if paredpreb == 0:
                return 93342
            elif paredpreb == 1:
                return 95342
        elif rooms == 3:
            if paredpreb == 0:
                if pisonotiene == 0:
                    return 122383
                elif pisonotiene == 1:
                    return 65000
            elif paredpreb == 1:
                return 97795
        elif rooms == 4:
            if paredpreb == 0:
                return 126042
            elif paredpreb == 1:
                return 101857
        elif rooms == 5:
            if paredpreb == 0:
                return 157052
            elif paredpreb == 1:
                return 137609
        elif rooms == 6:
            if paredpreb == 0:
                return 230395
            elif paredpreb == 1:
                return 93108
        elif rooms == 7:
            return 287232
        elif rooms == 8:
            if paredpreb == 0:
                return 434370
            elif paredpreb == 1:
                return 200000
        elif rooms == 9:
            return 605221
        elif rooms == 10:
            return 570540
        else:
            return 600000
    return v2a1

#here we will apply the impute function on train data and test data
datatrain['v2a1'] = datatrain[['v2a1','rooms','paredpreb','pisonotiene']].apply(impute_v2a1, axis = 1)
datatest['v2a1'] = datatest[['v2a1','rooms','paredpreb','pisonotiene']].apply(impute_v2a1, axis = 1)

def impute_meaneduc(coulmns):
    meaneduc = coulmns[0]
    instlevel9 = coulmns[1]
    if pd.isna(meaneduc):
        if instlevel9 == 0:
            meaneduc = 9.121911
        elif instlevel9 == 1:
            meaneduc = 16.244444
    return meaneduc

datatrain['meaneduc'] = datatrain[['meaneduc','instlevel9']].apply(impute_meaneduc, axis=1)
datatest['meaneduc'] = datatest[['meaneduc','instlevel9']].apply(impute_meaneduc, axis=1)

#third coulmn is 'SQBmeaned' which is the square root for 'meaneduc'
datatrain['SQBmeaned'].replace(np.NaN,datatrain['meaneduc']**2, inplace = True)
datatest['SQBmeaned'].replace(np.NaN,datatest['meaneduc']**2, inplace = True)

#forth coulmn is 'v18q1' which is dependt on 'v18q' coulmn 
#dealing with that column by puting the NaN values => Zeros
datatrain['v18q1'].replace(np.NaN,0,inplace=True)
datatest['v18q1'].replace(np.NaN,0,inplace=True)

#Last column
def impute_rez_esc(coulmns):
    rez_esc = coulmns[0]
    age = coulmns[1]
    escolari = coulmns[2]
    if pd.isna(rez_esc):
        if age < 7:
            return 0
        else:
            return age-escolari-7
    return rez_esc


datatrain['rez_esc'] = datatrain[['rez_esc','age','escolari']].apply(impute_rez_esc, axis=1)
datatest['rez_esc'] = datatest[['rez_esc','age','escolari']].apply(impute_rez_esc, axis=1)

#here we will save the 'Id' before droping it, because we want use it in submission file
y_id = datatest.iloc[:,0]

#Droping unwanted coulmns
datatrain.drop(['Id','idhogar'], axis = 1, inplace = True)
datatest.drop(['Id','idhogar'], axis = 1, inplace = True)

#replace no=>0, yes=>1
datatrain['dependency'].replace('no',0, inplace = True)
datatrain['dependency'].replace('yes',1, inplace = True)
datatrain['edjefa'].replace('no',0, inplace = True)
datatrain['edjefa'].replace('yes',1, inplace = True)
datatrain['edjefe'].replace('no',0, inplace = True)
datatrain['edjefe'].replace('yes',1, inplace = True)

#replace no=>0, yes=>1
datatest['dependency'].replace('no',0, inplace = True)
datatest['dependency'].replace('yes',1, inplace = True)
datatest['edjefa'].replace('no',0, inplace = True)
datatest['edjefa'].replace('yes',1, inplace = True)
datatest['edjefe'].replace('no',0, inplace = True)
datatest['edjefe'].replace('yes',1, inplace = True)

#Split training data
x_train = datatrain.iloc[:,:-1]
y_train = datatrain.iloc[:,-1]

#for that reason we will change the type from object to float
x_train.iloc[:,98:99] = x_train.iloc[:,98:99].astype(str).astype(float)
x_train.iloc[:,99:100] = x_train.iloc[:,99:100].astype(str).astype(float)
x_train.iloc[:,100:101] = x_train.iloc[:,100:101].astype(str).astype(float)

datatest.iloc[:,98:99] = datatest.iloc[:,98:99].astype(str).astype(float)
datatest.iloc[:,99:100] = datatest.iloc[:,99:100].astype(str).astype(float)
datatest.iloc[:,100:101] = datatest.iloc[:,100:101].astype(str).astype(float)


#trying xgboost machine learning algorithm for training
from xgboost import XGBClassifier
classifier = XGBClassifier(max_depth=100,n_estimators=500, learning_rate=1.0)
classifier.fit(x_train,y_train)

#Now the algorithm will predict the Target
y_pred = classifier.predict(datatest)

#create submission file wich contains 'Id' and 'Target' that was predicted
sbmt = pd.DataFrame({'Id':y_id, 'Target':y_pred})

sbmt.to_csv('submission.csv',index=False)

#Finish the project
