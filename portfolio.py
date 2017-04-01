import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class portfolio():
    
    def __init__(self, retv, riskv, corm):
        ''' initialization of the portfolio management object 
            - retv - expected return vector
            - risk - risk vector
            - corm - correlation matrix
        '''
        self.retv = retv
        self.riskv = riskv
        self.corm = corm
        
        # compute the covariance matrix
        self.covm = np.dot(np.dot(np.diag(self.riskv), self.corm), np.diag(self.riskv))
        
        # number of stocks in portfolio
        self.n = self.retv.shape[0]
   
    
    def frontier(self, expret):
        ''' computes the risk for a given expected return
            - expret - expected returns
        '''

        def minarg(w):
            ''' objective function for minimization procedure'''
            return np.dot(w, np.dot(self.covm, w))
    
        cons = [
            {'type': 'eq', 'fun':lambda x: 1 - np.sum(x)},                     # sum of weights is 1
            {'type': 'eq', 'fun':lambda x: expret - np.dot(self.retv, x)}      # expected return
        ]

        #bounds, weights must be within 0 and 1
        bnds = tuple((0,1) for x in range(self.n))
        
        #initial weight guess 
        winit = np.ones([self.n,1]) / self.n
        
        # compute weights
        w = minimize(minarg, winit, method='SLSQP',bounds=bnds, constraints=cons)
        
        return w.x

    
    def sharpe(self):
        '''computes the weights for the sharpe point'''
        
        def minargsharp(w):
            '''objective function for minimization procedure'''
            exprsk = np.dot(w, self.retv)
            var = np.dot(np.dot(w, self.covm), w)
            return -exprsk/np.sqrt(var)

        # define constraints and bounds
        cons = [{'type': 'eq', 'fun':lambda x: 1 - np.sum(x)}]
        bnds = tuple((0,1) for x in range(self.n))
        winit = np.ones([self.n,1]) / self.n
        
        # minimization procedure
        w = minimize(minargsharp, winit, method='SLSQP', bounds=bnds, constraints=cons)
        
        return w.x
    
    
    def plotfrontier(self, xlim=[0, 1], ylim=[0, 0.250], stocknames=''):
        '''plots the frontier and the maximum sharpe portfolio
              xlim, ylim - axis limits to be passed to the plot function
              stocknames - names of the drawn stocks
        '''
        plt.figure()
        ax = plt.subplot()
        x = []
        y = []

        for i in np.linspace(self.retv.min(), self.retv.max(), num=10):
            w = self.frontier(i)
            rettot = np.dot(w, self.retv)
            var = np.dot(np.dot(w, self.covm), w)
            x.append(np.sqrt(var))
            y.append(rettot)

        # plot the frontier with lines and dots and add values
        plt.scatter(x, y, alpha=0.3)
        plt.plot(x, y, alpha=0.3)
        for i, j in zip(x, y):
            ax.annotate("{:.2f}".format(j/i), xy=(i-0.02,j), horizontalalignment='center', verticalalignment='top', alpha= 0.3)

        # plot the stocks
        plt.scatter(self.riskv, self.retv, marker='+', s=200)
        if stocknames != '':
            cnt = 0 
            for i, j in zip(self.riskv, self.retv):
                ax.annotate(stocknames[cnt], xy=(i+0.02,j+0.005), horizontalalignment='center', verticalalignment='top')
                cnt = cnt + 1

        # plot the sharpe portfolio
        w = self.sharpe()
        var = np.dot(np.dot(w, self.covm), w)
        plt.scatter(np.sqrt(var), np.dot(w, self.retv), marker='o', s=100, c='r')
        plt.xlim(xlim)
        plt.ylim(ylim)

        plt.xlabel('Risk')
        plt.ylabel('Return')
        plt.title('Efficient frontier')

        plt.show()
        

'''executing this file as a script'''
if __name__ == "__main__":
  pass
