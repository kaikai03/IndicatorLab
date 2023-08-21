
import sys
import os 
module_path = os.path.abspath(os.path.join('..')) 
if module_path not in sys.path: 
    sys.path.append(module_path)

import numpy as np
import pandas as pd

import QUANTAXIS as QA

import tools.Sample_Tools as smpl
from base.JuUnits import excute_for_multidates

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from numba import jit

@jit(nopython=True)
def calc_deviation(MACD_origin_rolldata, window):
    x = MACD_origin_rolldata
    if x.shape[0] < window:
        return np.nan,np.nan
    x_t = x.T
    MACD = x_t[0]
    origin = x_t[1]
    if MACD is None or origin is None:
        return np.nan,np.nan
    if MACD is np.nan or origin is np.nan:
        return np.nan,np.nan
    
    last_MACD = np.argsort(MACD)[-1]
    last_origin = np.argsort(origin)[-1]

    diff = 0.0
    if last_origin < 1.0:#新低，底背离
        diff = np.abs(last_MACD - last_origin)
        
    if last_origin > 8.0:#新高,顶背离
        diff =  np.abs(last_MACD - last_origin)*-1.0
    return 0,diff



def MACD_JCSC(data_series,SHORT=12,LONG=26,M=9,deviate_window=10):    
    def kerrel(stock_single):
        DIFF = QA.EMA(stock_single,SHORT) - QA.EMA(stock_single,LONG)
        DEA = QA.EMA(DIFF,M)
        MACD = 2*(DIFF-DEA)

        CROSS_JC = QA.CROSS(DIFF,DEA)
        CROSS_SC = QA.CROSS(DEA,DIFF)
        CROSS = CROSS_JC + CROSS_SC*-1
        
        tmp_df = pd.DataFrame({'MACD':MACD,'origin_data':stock_single},columns=['MACD','origin_data'])
        deviation = tmp_df.dropna().rolling(deviate_window, method='table').apply(lambda x:calc_deviation(x,deviate_window), raw=True, engine='numba')['origin_data']

        
        return pd.DataFrame({'MACD':MACD,'DIFF':DIFF,'DEA':DEA,'MACD_CROSS':CROSS,'DEVIATE':deviation})

    return excute_for_multidates(data_series, kerrel, level=1)


def MACD_plot(MACD_df,low_frequence=True):
    groups = MACD_df.groupby(level=1)
    assert len(groups)<20,'不允许超过20组'
    fig = plt.figure(figsize=(2120/72,220*len(groups)/72))
    for idx,item in enumerate(groups):
        inds_ = item[1].reset_index('code',drop=True)
        
        ax = fig.add_subplot(len(groups),1,idx+1)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        
        ##axis不转成字符串的话，bar和line的x轴有时候对不上，原因未知
        formater = '%Y%m%d' if low_frequence else '%Y%m%d %H%M%S'
        index_ = [pd.to_datetime(x).strftime(formater) for x in inds_.index.values]
        #d = item[1].reset_index(('date','code'),drop=True)
        ax.set_title(item[0],color='r', loc ='left', pad=-10) 
        if 'DIFF' in MACD_df.columns and 'DEA' in MACD_df.columns: 
            DD = inds_[['DIFF','DEA']]
            DD.index = index_
            DD.plot(kind='line', ax=ax)
        macd = inds_['MACD']
        ax.bar(index_,macd.values)
        plt.xticks(rotation = 0)
    
