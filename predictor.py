import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from singlestock import preprocessdata

def customscore(y, ypred):    
    return r2_score(y, ypred)


class stockprediction():
    '''
       implements supervised regression learning algorithms
    '''
    def __init__(self, symbol, lookbackweeks=4, alldata=False):
        
        # define predictors of interest
        self.preds = [
            SVR(C=1, kernel='linear'),
            RandomForestRegressor(n_estimators=80)
        ]
        
        # get data
        self.data = preprocessdata(symbol).getrollingvars()
        self.lookback = lookbackweeks * 5
        
        self.alldata=alldata
        
        # get train/test vectors
        [self.xtrain, self.ytrain, self.xtest, self.ytest] = self.__gettraintest(self.data, ratio=0.8, nweeks=0)
        
    
    def __createcummulativex(self, data):
        '''concatenates columns in lookback interval'''

        if self.alldata:
            res = data.copy()
        else:
            res = data[['Adj_Close']].copy()

        keys = res.keys()
        
        for cnt in range(1, self.lookback):
            for s in keys:
                res[s + str(cnt)] = data[s].shift(cnt).copy()
               
        res = res.iloc[self.lookback-1:]
        
        return res

    def __gettraintest(self, mydata, ratio=0.8, nweeks=0):
        '''
        returns proper train and test vectors for estimators
        
            nweeks - number of weeks ahead for the prediction
                if nweeks = 0 - one day look ahead
            lookbackweeks - number of weeks for lookback interval
            ratio - how many data points allocated for training
        '''
        lookback = self.lookback
    
        # compute how many points ahead we need to look
        offset = nweeks * 5
        if offset == 0:
            offset = 1

        # test if we have enough data
        if mydata.shape[0] < offset + lookback:
            print("error: not enough training data points given")
            return
    
        # make a copy, sort and extract x and y
        tx = mydata.copy()
    
        # test if we have enough data
        if tx.shape[0] < offset + lookback:
            print("error: not enough training data points given")
            return

        index = int((tx.shape[0]-lookback-offset) * ratio) + lookback + offset
        ty = tx['Adj_Close'].copy()
   
        # create x vectors
        xtrain = self.__createcummulativex(tx.iloc[:index-offset])
        xtest = self.__createcummulativex(tx.iloc[index-lookback-offset+1:-offset])
    
        # create y vectors    
        ytrain = ty.iloc[lookback + offset - 1:index]
        ytest = ty.iloc[index:]
        
        # force the same index to prevent awkward behavior
        xtrain.set_index(ytrain.index, inplace = True)
        xtest.set_index(ytest.index, inplace = True)
    
        return [xtrain, ytrain, xtest, ytest]


    def searchbest(self, pid=0):
        '''finds the best estimator'''
        if pid==0:
            param_grid = [{'C': [0.5, 0.75, 1, 1.5, 2, 5, 10], 'kernel': ['linear', 'rbf', 'sigmoid']}]
        if pid==1:
            param_grid = [{'n_estimators': [200], 'min_samples_split': [2,3,5,10], 'min_samples_leaf': [2,5, 10]}]
          
        fscore = make_scorer(customscore, greater_is_better = True)
          
        cv = TimeSeriesSplit(n_splits=5)
        gs = GridSearchCV(self.preds[pid], param_grid, fscore, n_jobs=4, cv=cv)
        gs.fit(self.xtrain, self.ytrain)
        self.preds[pid] = gs.best_estimator_
        print "parameters found:", gs.best_params_
        
        return self.preds[pid]

    def train(self, pid=0):
        '''train interface for the estimators'''
        self.preds[pid].fit(self.xtrain, self.ytrain)
    
    
    def query(self, xin, yin, pid=0):
        '''query interface for the estimators'''
        
        pred = self.preds[pid]
        ypred = pred.predict(xin)
        score = pred.score(xin, yin) 
        corr  = np.corrcoef(yin, ypred)
        
        # transform returned values into a pandas series
        ypred = pd.Series(ypred, index=yin.index)

        return [ypred, score, corr]
        
        

'''executing this file as a script'''
if __name__ == "__main__":
    pass