
import numpy as np
import pandas as pd

import QUANTAXIS as QA
from QUANTAXIS.QAUtil.QAParameter import FREQUENCE

import Ind_Model_Base

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class OBV(Ind_Model_Base.Ind_Model):
    '''能量潮指标改进版'''
    def __init__(self,data, frequence=FREQUENCE.DAY):
        super().__init__(data, 'OBV_EX', frequence)
    
    def on_set_params_default(self):
        return {}
        
    def on_indicator_structuring(self, data):
        #return data.add_func(self.MACD_JCSC,**self.pramas)
        return self.excute_for_multicode(data, self.OBV)

    
#     def on_desition_structuring(self, data, ind_data):
#         return None
        
    def OBV(self,dataframe):
        '''多空比率净额= [（收盘价－最低价）－（最高价-收盘价）] ÷（ 最高价－最低价）×V'''
        long_short_ratio=((dataframe.close - dataframe.low) - (dataframe.high - dataframe.close)) / (dataframe.high-dataframe.low) * dataframe.volume

        return pd.DataFrame({'LSR':long_short_ratio})
    
    
    def plot(self,figsize=(16,6)) -> dict:
        fig = plt.figure(figsize=figsize)
        groups = self.ind_df.groupby(level=1)
        for idx,item in enumerate(groups):
            inds_ = item[1].reset_index('code',drop=True)
            ax = fig.add_subplot(len(groups),1,idx+1)
            inds_.plot(ax=ax)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
            plt.xticks(rotation = 0)
            plt.legend([item[0]])
    
    def plot_mix(self,figsize=(16,6)) -> dict:
        fig = plt.figure(figsize=figsize)
        groups = self.ind_df.groupby(level=1)
        groups = self.ind_df.groupby(level=1)
        codes = []
        for idx,item in enumerate(groups):
            inds_ = item[1].reset_index('code',drop=True)
            plt.plot(inds_)
            codes.append(item[0])
        plt.legend(codes)
    
    