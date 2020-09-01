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

RSI=True
RSIOverBought=70
RSIOverSold=30


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
        
        sys.stdout = sys.__stdout__
        return data

    def find_anomalies(self, data):
        global STD_CUTTOFF
        indexs = []
        outliers = []
        #printing data for analysis. delete later
        pd.set_option('display.max_rows',999)
        print(data)

        data_std = np.std(data['Volume'])
        data_mean = np.mean(data['Volume'])
        anomaly_cut_off = data_std * STD_CUTTOFF
        upper_limit = data_mean + anomaly_cut_off
        data.reset_index(level=0, inplace=True)
        for i in range(len(data)):
            temp = data['Volume'].iloc[i]
            if temp > upper_limit:
                indexs.append(str(data['Date'].iloc[i])[:-9])
                outliers.append(temp)
        d = {'Dates': indexs, 'Volume': outliers}
        return d

    def customPrint(self, d, tick):
        print("\n\n\n*******  " + tick.upper() + "  *******")
        print("Ticker is: "+tick.upper())
        for i in range(len(d['Dates'])):
            str1 = str(d['Dates'][i])
            str2 = str(d['Volume'][i])
            print(str1 + " - " + str2)
        print("*********************\n\n\n")

    def days_between(self, d1, d2):
        d1 = datetime.datetime.strptime(d1, "%Y-%m-%d")
        d2 = datetime.datetime.strptime(d2, "%Y-%m-%d")
        return abs((d2 - d1).days)

    def parallel_wrapper(self, x, currentDate, positive_scans):
        
        global DAY_CUTTOFF
        d = (self.find_anomalies(self.getData(x)))

        if d['Dates']:
            for i in range(len(d['Dates'])):
                if self.days_between(str(currentDate)[:-9], str(d['Dates'][i])) <= DAY_CUTTOFF:
                    self.customPrint(d, x)
                    stonk = dict()
                    stonk['Ticker'] = x
                    stonk['TargetDate'] = d['Dates'][0]
                    stonk['TargetVolume'] = str(
                        '{:,.2f}'.format(d['Volume'][0]))[:-3]
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

    def main_func(self):
        print("starting")
        """
        pd.set_option('display.max_rows',999)
        testticker = yf.Ticker("AAPL")
        tickerhist = testticker.history(period="2y")

        tickerhist['50MA'] = tickerhist.Close.rolling(50).mean()
        tickerhist['200MA'] = tickerhist.Close.rolling(200).mean()
        print(tickerhist[-256:])

        
        print out the whole row in pandas
        info = sorted([[k,v] for k,v in testticker.info.items()])
        for k,v in info:
            print(f'{k} : {v}')
        """

        StocksController = NasdaqController(True)
        print("stockController True")
        list_of_tickers = StocksController.getList()
        print("got stock list")
        currentDate = datetime.datetime.strptime(
            date.today().strftime("%Y-%m-%d"), "%Y-%m-%d")
        start_time = time.time()
        print("starting manager")
        manager = multiprocessing.Manager()
        print("process manager")
        positive_scans = manager.list()
        print("while loop")
        with parallel_backend('loky', n_jobs=multiprocessing.cpu_count()):
            Parallel()(delayed(self.parallel_wrapper)(x, currentDate, positive_scans)
                       for x in tqdm(list_of_tickers))

        print("\n\n\n\n--- this took %s seconds to run ---" %
              (time.time() - start_time))

        return positive_scans


if __name__ == '__main__':
    mainObj().main_func()
