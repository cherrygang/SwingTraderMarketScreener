import os
import time
import yfinance as yf
import dateutil.relativedelta
from datetime import date
import datetime
import numpy as np
import sys
from stocklist import NasdaqController
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
import lxml
import pandas as pd

###########################
# THIS IS THE MAIN SCRIPT #
###########################

# Change variables to your liking then run the script
MONTH_CUTTOFF = 12
DAY_CUTTOFF = 1
STD_CUTTOFF = 2

#Number of days to look back
LOOKBACK = 3

#RSIunder = RSI under RSIOverSold number
#RSIover = RSI over RSIOverBought number
#MACD = MACD breaking out over 9 day EMA
#MAbuy = 9 day EMA breaking out over 20 day EMA
#MAsell = 9 day EMA crossing under 20 day EMA
#SCREENS = [RSIunder, RSIover, MACD, MAbuy, MAsell] (1 = turned on, 0 = turned off)
SCREENS = [1, 0, 0, 1, 0]

#RSI Info
RSIunder=True
RSIOverSold=30

RSIover=False
RSIOverBought=70

#Turns on screen for MACD (i.e. MACD cross 9d EMA)
MACD=False

#Turns on screen for Moving Average (9 day EMA cross 20 day EMA). Buy & Sell controls for buy & sell signals
MAbuy=True
MAsell=False


class mainObj:

    def __init__(self):
        pass

    def getData(self, ticker):
        global MONTH_CUTOFF
        currentDate = datetime.date.today() + datetime.timedelta(days=1)
        pastDate = currentDate - \
            dateutil.relativedelta.relativedelta(months=MONTH_CUTTOFF)
        sys.stdout = open(os.devnull, "w")
        data = yf.download(ticker, pastDate, currentDate)
        #add 50 & 200 moving day averages to data from yfinance
        data['50MA'] = data.Close.rolling(50).mean()
        data['200MA'] = data.Close.rolling(200).mean()
        data['RSI'] = self.getRSI(data['Close'],14)
        data['MACD'] = self.getMACD(data['Close'])
        data['9dEMA'] = self.getEMA(data['Close'],9)
        data['20dEMA'] = self.getEMA(data['Close'],20)
        data['test1'] = 0
        data['test2'] = 2
        
        sys.stdout = sys.__stdout__
        return data

    def screener(self, data):
        global STD_CUTTOFF
        flag = 0
        outliers = []
        addInfo=""
        #printing data for analysis. delete later
        #pd.set_option('display.max_rows',999)
        #print(data)


        data.reset_index(level=0, inplace=True)


        #RSI analysis
        if (RSIover or RSIunder):

            for i in data['RSI'].iloc[-3:]:
                if RSIunder:
                    if i < 30:
                        #[:-9] at the end is to remove lines & spaces from 
                        addInfo = addInfo + "\nRSI: "+ str(round(i)) +" Over Sold"
                        flag= 1
                        break
                if RSIover:
                    if i > 70:
                        addInfo = addInfo + "\nRSI: "+ str(round(i)) +" Over Bought"
                        flag= 1
                        break
        #MACD analysis
        if MACD:
            if self.lineCross(data['MACD'],data['9dEMA'],3):
                addInfo = addInfo + "\nMACD cross " 
                flag +=1
                
        #MA analysis

        if MAbuy:
            if self.lineCross(data['9dEMA'],data['20dEMA'],3):
                addInfo = addInfo + "\n9 day EMA & 20 day EMA Buy Signal"
                flag +=1
        if MAsell:
            if self.lineCross(data['20dEMA'],data['9dEMA'],3):
                addInfo = addInfo + "\n9 day EMA & 20 day EMA Sell Signal"
                flag +=1
        d = {'flag': flag, 'Volume': outliers, 'Info': addInfo}
        return d

    def customPrint(self, d, tick):
        print("\n\n\n*******  " + tick.upper() + "  *******")
        print("Ticker is: "+tick.upper())

        print("\n "+d['Info'])
        print("*********************\n\n\n")

    def days_between(self, d1, d2):
        d1 = datetime.datetime.strptime(d1, "%Y-%m-%d")
        d2 = datetime.datetime.strptime(d2, "%Y-%m-%d")
        return abs((d2 - d1).days)

    def parallel_wrapper(self, x, currentDate, positive_scans):
        
        global DAY_CUTTOFF
        d = (self.screener(self.getData(x)))
        if d['flag'] == 2 :
            self.customPrint(d,x)

            stonk = dict()
            stonk['Ticker'] = x
            positive_scans.append(stonk)

    def getRSI(self, data, time_window):
        diff = data.diff(1).dropna()
        gain_chg= 0 * diff
        loss_chg = 0 * diff
        gain_chg[diff > 0] = diff[diff > 0]
        loss_chg[diff < 0] = diff[diff < 0]
        gain_chg_avg = gain_chg.ewm(com=time_window-1, min_periods=time_window).mean()
        loss_chg_avg = loss_chg.ewm(com=time_window-1, min_periods=time_window).mean()

        rs = abs(gain_chg_avg/loss_chg_avg)
        rsi = 100-100/(1+rs)
        return rsi

    def getMACD(self, data):
        return self.getEMA(data,12)-self.getEMA(data,26)

    def getEMA(self,data,time_frame):
        return data.ewm(span=time_frame,adjust=False).mean()

    def lineCross(self, line1, line2, lookback):
        #if line1 is initially lower than line 2 & cross then return True
        try:
            if (line1.iloc[-lookback]-line2.iloc[-lookback]<0):
                if (line1.iloc[-1]-line2.iloc[-1]>0):
                    return True
                return False
        except IndexError:
            return False


    def main_func(self):
        StocksController = NasdaqController(True)
        list_of_tickers = StocksController.getList()
        currentDate = datetime.datetime.strptime(
            date.today().strftime("%Y-%m-%d"), "%Y-%m-%d")
        start_time = time.time()
        manager = multiprocessing.Manager()
        positive_scans = manager.list()
        with parallel_backend('loky', n_jobs=multiprocessing.cpu_count()):
            Parallel()(delayed(self.parallel_wrapper)(x, currentDate, positive_scans)
                       for x in tqdm(list_of_tickers))

        print("\n\n\n\n--- this took %s seconds to run ---" %
              (time.time() - start_time))

        return positive_scans


if __name__ == '__main__':
    mainObj().main_func()
