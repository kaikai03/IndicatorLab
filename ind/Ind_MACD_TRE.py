
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

from matplotlib.collections import LineCollection



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


def get_swing_band (MACD,single_stock_df,plot=False,DIFF=None,DEA=None):
    variant = single_stock_df.close
    MACD_DIF = MACD/2
    
    # 𝑇𝑅[𝑡] = max{最高价[t] − 最低价[t], 最高价[t] − 收盘价[t − 1], 收盘价[t − 1] − 最低价[t]}
    HL = single_stock_df.high - single_stock_df.low
    close_last = single_stock_df.close.shift(1)
    HC = single_stock_df.high - close_last
    CL = close_last - single_stock_df.low
    con = HL < HC
    HL[con]=HC[con]
    𝑇𝑅 = HL
    con = 𝑇𝑅 < CL
    𝑇𝑅[con] = CL[con]
    𝐴𝑇𝑅 = TR.rolling(100).mean()

    cum_value=[0.00001]
    def calc_intergral(x,cum):
        if x is np.nan:
            return x
        if np.sign(cum[0]) == np.sign(x):
            cum[0] +=x
        else:
            cum[0] = x
        return cum[0]
    

    # 同号差异累计值
    intergral = MACD_DIF.apply(lambda x: calc_intergral(x,cum_value))
    thres = .5
    delta = 𝐴𝑇𝑅 * thres
    intergral_origin = pd.Series(np.nan,index = intergral.index)
    intergral_origin[intergral >= delta] = 1
    intergral_origin[intergral <= delta*-1] = -1
    intergral_origin[𝐴𝑇𝑅.isna()]=np.nan
    intergral_filled= intergral_origin.fillna(method='ffill')

    if plot:
        if DIFF is None or DEA is None:
            raise 'if want to plot, pass DIFF and DEA'
        fig = plt.figure(figsize=(2820/72/2,420/72))
        colors = pd.Series('black',index=variant.index)
        colors[intergral_filled>0]='red'
        colors[intergral_filled<0]='green'
        ind = range(len(variant))
        xy = pd.DataFrame({"X":ind,"Y":variant},index=variant.index).values.reshape(-1, 1, 2)
        segments = np.hstack([xy[:-1], xy[1:]])
        coll = LineCollection(segments, color=colors)
        coll.set_array(np.random.random(xy.shape[0]))
        ax = fig.gca()
        ax.add_collection(coll)
        ax.autoscale_view()

        ax2 = plt.gca().twinx()
        DIFF.plot(ax=ax2)
        DEA.plot(ax=ax2)
        intergral_filled.plot(ax=ax2,color='grey',linestyle=":")

    return intergral_filled





def MACD_JCSC(stock_df,main_column='close',SHORT=12,LONG=26,M=9,deviate_window=10):
    def kerrel(single_stock_df):
        main_variant = single_stock_df[main_column]
        DIFF = QA.EMA(main_variant,SHORT) - QA.EMA(main_variant,LONG)
        DEA = QA.EMA(DIFF,M)
        MACD = 2*(DIFF-DEA)

        CROSS_JC = QA.CROSS(DIFF,DEA)
        CROSS_SC = QA.CROSS(DEA,DIFF)
        CROSS = CROSS_JC + CROSS_SC*-1
        
        tmp_df = pd.DataFrame({'MACD':MACD,'main_variant':main_variant},columns=['MACD','main_variant'])
        deviation = tmp_df.dropna().rolling(deviate_window, method='table').apply(lambda x:calc_deviation(x,deviate_window), raw=True, engine='numba')['main_variant']

        swing_band = get_swing_band(MACD,single_stock_df)
        
        return pd.DataFrame({'MACD':MACD,'DIFF':DIFF,'DEA':DEA,'MACD_CROSS':CROSS,'DEVIATE':deviation,'SWING_BAND':swing_band})

    return excute_for_multidates(stock_df, kerrel, level=1)


def MACD_plot(MACD_df,low_frequence=True):
    groups = MACD_df.groupby(level=1)
    assert len(groups)<20,'一次显示不允许超过20组'
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
    
