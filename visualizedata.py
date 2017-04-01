import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os.path
import seaborn
from yahoo_finance import Share



def retrievedata(symbol, datestart="2011-01-01", dateend="2017-01-01"):
    '''function retrieves historical data for a given company
       symbol - company symbol
       datestart - starting date
       dateend - ending date
    '''
  
    # create file name
    fname = symbol + datestart + dateend + '.pickle'
    
    # check if data exists already
    if os.path.isfile('data/' + fname):
        print symbol + " - data found locally"
        df = pd.read_pickle('data/' + fname)
    else:
        print symbol + " - getting data from yahoo finance"
        temp = Share(symbol)
        data = temp.get_historical(datestart, dateend)
  
        # convert to data frame
        df = pd.DataFrame(data)
        df.set_index(pd.to_datetime(df['Date']), inplace = True)
        # drop all columns except Adj_Close
        df = df[['Adj_Close']]
        df = df.apply(pd.to_numeric)
        # sort the index ascending
        df.sort_index(inplace = True)
        
        # save dataframe to file
        df.to_pickle('data/' + fname)
  
    return df


  
def visualize(symbols, scaled=True, title=""):
    '''function visualizes a portfolio of data over a period of time
    '''
    plt.figure(figsize=(16,6))

    dframes = []
    for s in symbols:
        df = pd.read_pickle('data/' + s + '2011-01-012017-01-01.pickle')
        if scaled:
            df['Close_Norm'] = df['Adj_Close'] / df['Adj_Close'].iloc[0]
            df['Close_Norm'].plot(label=s)
        else:
            df['Adj_Close'].plot(label=s)

    # fix the x tick values and display
    ax = plt.gca()
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 90))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))

    plt.title(title)
    plt.legend(loc=2)
    plt.show()



def printstats(symbols):
    '''helper function to compute data summaries and kurtosis and display a few histograms
    '''
    res = pd.DataFrame()
    for s in symbols:
        df = pd.read_pickle('data/' + s + '2011-01-012017-01-01.pickle')
        temp = df.describe()
        temp['Adj_Close']['kurtosis'] = df['Adj_Close'].kurtosis()
        res[s] = temp['Adj_Close']

        # plot a couple of histograms
        if s=='AMZN' or s=='YHOO':
            plt.figure(figsize=(8,6))
            df['Adj_Close'].hist(bins=20)
            plt.title('Histogram for ' + s)
            plt.xlabel('Stock price')
            plt.show()

    print res
    res.to_csv('res.csv')


'''the symbols of interest for this project'''
symbols = [
  'AAPL', 
  'AMZN', 
  'GOOG', 
  'MSFT', 
  'YHOO',
  'SPY'
]


'''executing this file as a script'''
if __name__ == "__main__":
  pass
