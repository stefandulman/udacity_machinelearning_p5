import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from visualizedata import retrievedata


class preprocessdata():
    '''adds new columns to data and makes it ready for the estimators'''
    
    def __init__(self, symbol, data=None):
        '''constructor - if data is provided, no loading or yahoo finance involved'''
        self.symbol = symbol
        if data is None:
            # get data from local storage or yahoo finance
            self.data = retrievedata(symbol)
        else:
            self.data = data

        # scale the data
        self.data['Adj_Close'] = self.data['Adj_Close'] / self.data['Adj_Close'].iloc[0]
        

    def getbollinger(self, nsigmas=2, nweeks=12, dropclose=False, ysignal='dummy'):
        '''compute the bollinger bands
             dropclose - boolean value signaling if 'Adj_Close' should be dropped from the result
             nsigmas - band width in +-sigmas. default 2
             nweeks - number of weeks for lookback history
             ysignal - signal for which to generate the buy/sells signals
        '''
        td = self.data.copy()
        td['Bollinger Mean'] = td['Adj_Close'].rolling(nweeks*5).mean()
        td['Bollinger Std']  = td['Adj_Close'].rolling(nweeks*5).std()
        td['Bollinger Bottom'] = td['Bollinger Mean'] - td['Bollinger Std'] * nsigmas
        td['Bollinger Top']    = td['Bollinger Mean'] + td['Bollinger Std'] * nsigmas
        
        td['Bollinger Buy']  = td[ysignal] < td['Bollinger Bottom']
        td['Bollinger Sell'] = td[ysignal] > td['Bollinger Top']

        # drop the adj_close column
        td.drop(['Adj_Close'], axis=1, inplace=True)
                
        if dropclose:
            td.drop(['Bollinger Buy', 'Bollinger Sell', 'Bollinger Bottom', 'Bollinger Top'], axis=1, inplace=True)
            
        return td
    
    
    def getdualcrossover(self, nweeksshort=2, nweekslong=12, dropclose=False):
        '''compute a slow and a fast moving average
             nweeksshort - number of weeks for short-term moving average
             nweekslong - number of weeks for long-term moving average
             dropclose - boolean value signaling if 'Adj_Close' should be dropped from the result
        '''
        td = self.data.copy()
        td['Short Term Average'] = td['Adj_Close'].rolling(nweeksshort*5).mean()
        td['Long Term Average'] = td['Adj_Close'].rolling(nweekslong*5).mean()
        
        td['Crossover Buy']  = td['Short Term Average'] < td['Long Term Average'] * 1.01
        td['Crossover Sell'] = td['Short Term Average'] > td['Long Term Average'] * 1.01
        
        # drop the adj_close column
        td.drop(['Adj_Close'], axis=1, inplace=True)
        
        if dropclose:
            td.drop(['Crossover Buy', 'Crossover Sell'], axis=1, inplace=True)
            
        return td
    
    
    def getchanneltrading(self, nweeks=12, dropclose=False, ysignal='dummy'):
        '''compute the channel trading buy sell points
             nweeks - number of days to look back
             ysignal - signal for which to generate the buy/sells signals
             dropclose - boolean value signaling if 'Adj_Close' should be dropped from the result
        '''
        # compute the means
        td = self.data.copy()
        td['Channel Low']  = td['Adj_Close'].shift(1).rolling(nweeks*5).min()
        td['Channel High'] = td['Adj_Close'].shift(1).rolling(nweeks*5).max()
    
        # compute the buy/sells points
        td['Channel Buy']  = td[ysignal] < td['Channel Low']* 1.01
        td['Channel Sell'] = td[ysignal] > td['Channel High'] * 1.01
        
        # drop the adj_close column
        td.drop(['Adj_Close'], axis=1, inplace=True)
        
        if dropclose:
            td.drop(['Channel Buy', 'Channel Sell'], axis=1, inplace=True)
            
        return td
            
    
    def getrollingvars(self, bolns=2, bolnw=12, dcnws=2, dcnwl=12, ctnw=12, dropclose=True, ysignal='Adj_Close'):
        '''returns a data frame filled with all variables needed for the estimator
            bolns = bollinger number of sigmas
            bolnw = bollinger number of weeks
            dcnws = dual crossover number of weeks short
            dcnwl = dual crossover number of weeks long
            ctnw  = channel trading number of weeks
            dropclose - boolean value signaling if 'Adj_Close' should be dropped from the result
            ysignal - signal for which to generate the buy/sells signals
        '''
        td = self.data.copy()
        td = pd.concat([td, self.getbollinger(nsigmas=bolns, nweeks=bolnw, dropclose=dropclose, ysignal=ysignal)], axis=1)
        td = pd.concat([td, self.getdualcrossover(nweeksshort=dcnws, nweekslong=dcnwl, dropclose=dropclose)], axis=1)
        td = pd.concat([td, self.getchanneltrading(nweeks=ctnw, dropclose=dropclose, ysignal=ysignal)], axis=1)
        
        # add daily returns
        td['DR'] = td['Adj_Close'] / td['Adj_Close'].shift(1) - 1.0
        
        # drop the first lines which contain NANs due to rolling computations
        td.dropna(axis=0, inplace=True)
        
        return td



def filterduplicates(sell, buy):
    '''filters consecutive sell/buy signals'''
    # filter the predicted signals
    state = 0    # 0=none, -1=buy, 1=sell
    for i in range(0, buy.shape[0]):
        if sell.iloc[i]:
            if state != 1:
                state = 1
            else:
                sell.iloc[i] = False
        if buy.iloc[i]:
            if state != -1:
                state = -1
            else:
                buy.iloc[i] = False
    return [sell, buy]


    
def evalestim(p, symbol, x, y):
    ''' evaluate single stock buy/sells decissions on profit'''
    
    [ypred, score, corr] = p.query(x, y)
    
    # count the number of correct sell, buy signals vs the real situation
    buysig_real = (x['Adj_Close'] < y) & (x['Adj_Close'] < x['Adj_Close'].shift(1))
    buysig_pred = (x['Adj_Close'] < ypred) & (x['Adj_Close'] < x['Adj_Close'].shift(1))
    
    correct_buy = 1.0 * sum(buysig_real & buysig_pred) / sum(buysig_real)
    
    sellsig_real = (x['Adj_Close'] > y) & (x['Adj_Close'] > x['Adj_Close'].shift(1))
    sellsig_pred = (x['Adj_Close'] > ypred) & (x['Adj_Close'] > x['Adj_Close'].shift(1))
    
    correct_sell = 1.0 * sum(sellsig_real & sellsig_pred) / sum(sellsig_real)
    
    # filter the predicted signals
    [sellsig_pred, buysig_pred] = filterduplicates(sellsig_pred, buysig_pred)
    
    sales_real = sum(x['Adj_Close'][sellsig_real])
    sales_pred = sum(x['Adj_Close'][sellsig_pred])
    buy_real = sum(x['Adj_Close'][buysig_real])
    buy_pred = sum(x['Adj_Close'][buysig_pred])
    
    # buying decisions based on bollinger bands
    tx = x[['Adj_Close']].copy()
    tx['y'] = ypred
    tdata = preprocessdata(symbol, data=tx)
    tx = tdata.getrollingvars(dropclose=False, ysignal='y')
    
    # filter the bollinger and channel signals
    [tx['Bollinger Sell'], tx['Bollinger Buy']] = filterduplicates(tx['Bollinger Sell'], tx['Bollinger Buy'])
    [tx['Crossover Sell'], tx['Crossover Buy']] = filterduplicates(tx['Crossover Sell'], tx['Crossover Buy'])
    
    sales_bol = sum(tx['Adj_Close'][tx['Bollinger Sell']])
    sales_ct  = sum(tx['Adj_Close'][tx['Crossover Sell']])
    buy_bol = sum(tx['Adj_Close'][tx['Bollinger Buy']])
    buy_ct  = sum(tx['Adj_Close'][tx['Crossover Buy']])
    
    return [sales_real, sales_pred, buy_real, buy_pred, correct_buy, correct_sell, sales_bol, sales_ct, buy_bol, buy_ct]



class explorebuysell():
    '''visualizes the metrics for single stock decisions'''

    def __init__(self, symbol):
        ''' symbol - stock symbol
        '''
        self.ppdata = preprocessdata(symbol)
      

    def __filtercontdup(self, t):
        '''remove continuous duplicates
        '''
        # state indicates if we need to delete current point
        state = False
        for i in range(0, t.size):
            if state==True:
                if t[i] == True:
                    t[i] = False
                else:
                    state = False
            else:
                if t[i] == True:
                    state = True
        return t

        
    def __filterduplicates(self, dx, dy, ftype):
        ''' function takes two series of occurences and returns 
        a filtered version where duplicates in each series are 
        removed
        
            ftype - not - no filtering
                  - simple - only consecutive occurences are removed
                  - alternate - a buy signal will always be followed by a sell signal and opposite
        '''
        if ftype=='no':
            return [dx, dy]
        
        dx = self.__filtercontdup(dx)
        dy = self.__filtercontdup(dy)

        if ftype=='alternate':
            # state - marks which series had the last "true" value
            # -1 - first, 0 - none yet, 1 - second
            state = 0
        
            # choose one occurence after the other
            for i in range(0, dx.size):
            
                if dx[i]==True and state!=-1:
                    state = -1
                else:
                    dx[i] = False
                
                if dy[i]==True and state!=1:
                    state = 1
                else:
                    dy[i] = False
    
        return [dx, dy]
    

    def __plotbuysell(self, s, index_buy, index_sell, ftype):
        '''filter the indexes for multiple occurences
        '''
        index_buy, index_sell = self.__filterduplicates(index_buy, index_sell, ftype)

        if sum(index_buy) > 0:
            temp1 = s[index_buy].copy()
            temp1 = temp1.rename('Buy Decision')
            temp1.plot(style='ro')
    
        if sum(index_sell) > 0:
            temp2 = s[index_sell].copy()
            temp2 = temp2.rename('Sell Decision')
            temp2.plot(style='bo')    

            
    def bollinger(self, nsigmas=2, nweeks=12, ftype='no'):
        '''plots the bands
            nsigmas - band width in +-sigmas. default 2
            nweeks - number of weeks for lookback history
            ftype - filtering type on the points (none, simple, alternate)
        '''
        
        # get the data with bands
        td = self.ppdata.getbollinger(nsigmas, nweeks, dropclose=False, ysignal='Adj_Close')
        td['Adj_Close'] = self.ppdata.data['Adj_Close'].copy()
      
        # compute the buy/sells points
        index_buy  = td['Bollinger Buy'].copy()
        index_sell = td['Bollinger Sell'].copy()

        # plot the data
        plt.figure(figsize=(16,6))
        
        ax = td['Adj_Close'].plot()
        td['Bollinger Top'].plot(style='--', color=[0, 0, 1, 0.3])
        td['Bollinger Bottom'].plot(style='--', color=[1, 0, 0, 0.3])
        td['Bollinger Mean'].plot(style='--', color=[0, 0, 0, 0.3])
    
        self.__plotbuysell(td['Adj_Close'], index_buy, index_sell, ftype)
    
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 90))
    
        plt.legend()
        plt.title("Bollinger bands for " + self.ppdata.symbol + ' (nsigmas=' + str(nsigmas) + ', nweeks=' + str(nweeks) + ')')
        plt.show()
        

    def dualcrossover(self, nweeksshort=2, nweekslong=12, ftype='no'):
        '''plots the dual crossover lines
            nweeksshort - number of weeks for short-term moving average
            nweekslong - number of weeks for long-term moving average
        '''
        # get the data with the averages
        td = self.ppdata.getdualcrossover(nweeksshort, nweekslong, dropclose=False)
        td['Adj_Close'] = self.ppdata.data['Adj_Close'].copy()
    
        # compute the buy/sells points
        index_buy  = td['Crossover Buy'].copy()
        index_sell = td['Crossover Sell'].copy()

        # plot the data
        plt.figure(figsize=(16,6))
        
        ax = td['Adj_Close'].plot(title="Dual moving average crossover for " + self.ppdata.symbol + ' (nweeks_short=' + str(nweeksshort) + ', nweeks_long=' + str(nweekslong) + ')')
        td['Short Term Average'].plot(style='--', color=[1, 0, 0, 0.3])
        td['Long Term Average'].plot(style='--',  color=[0, 0, 1, 0.3])
    
        self.__plotbuysell(td['Adj_Close'], index_buy, index_sell, ftype)
    
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 90))
    
        plt.legend()
        plt.show()
        
    
    def channeltrading(self, nweeks=4, ftype='simple'):
        '''plots the dual crossover lines
            nweeks - number of days to look back
        '''
        # compute the means
        td = self.ppdata.getchanneltrading(nweeks, dropclose=False, ysignal='Adj_Close')
        td['Adj_Close'] = self.ppdata.data['Adj_Close'].copy()
    
        # compute the buy/sells points
        index_buy  = td['Channel Buy'].copy()
        index_sell = td['Channel Sell'].copy()

        # plot the data
        plt.figure(figsize=(16,6))
        
        ax = td['Adj_Close'].plot(title="Channel trading for " + self.ppdata.symbol + ' (nweks=' + str(nweeks) + ')')
        td['Channel Low'].plot(style='--', color=[1, 0, 0, 0.5])
        td['Channel High'].plot(style='--',  color=[0, 0, 1, 0.5])
    
        # filter the index for multiple occurences
        self.__plotbuysell(td['Adj_Close'], index_buy, index_sell, ftype)
    
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 90))
    
        plt.legend()
        plt.show()



'''executing this file as a script'''
if __name__ == "__main__":
    pass