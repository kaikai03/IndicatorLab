
import numpy as np
import pandas as pd

import QUANTAXIS as QA
import Ind_Model_Base

import matplotlib.pyplot as plt

class MACD_JCSC(Ind_Model_Base.Ind_Model):
    def __init__(self,data):
        super().__init__(data, 'MACD')

    
    def on_set_params_default(self):
        return {'SHORT':12,'LONG':26,'M':9}
        
    def on_indicator_structuring(self, data):
        #return data.add_func(self.MACD_JCSC,**self.pramas)
        return self.excute_for_multicode(data, self.MACD_JCSC, **self.pramas)

    
    def on_desition_structuring(self, data, ind_data):
        return pd.DataFrame({'res':ind_data['CROSS_JC'] + ind_data['CROSS_SC']*-1})
        
    def MACD_JCSC(self,dataframe,SHORT=12,LONG=26,M=9):
        print(SHORT,LONG,M)
        """
        1.DIF向上突破DEA，买入信号参考。
        2.DIF向下跌破DEA，卖出信号参考。
        """
        CLOSE=dataframe.close
        DIFF =QA.EMA(CLOSE,SHORT) - QA.EMA(CLOSE,LONG)
        DEA = QA.EMA(DIFF,M)
        MACD =2*(DIFF-DEA)

        CROSS_JC=QA.CROSS(DIFF,DEA)
        CROSS_SC=QA.CROSS(DEA,DIFF)
        ZERO=0
        return pd.DataFrame({'DIFF':DIFF,'DEA':DEA,'MACD':MACD,'CROSS_JC':CROSS_JC,'CROSS_SC':CROSS_SC,'ZERO':ZERO})
    
    def plot(self,figsize=(16,6)):
        fig = plt.figure(figsize=figsize)
        groups = self.ind_df.groupby(level=1)
        for idx,item in enumerate(groups):
            inds_ = item[1].reset_index('code',drop=True)
            ax = fig.add_subplot(len(groups),1,idx+1)
            index_ = [pd.to_datetime(x).strftime('%Y%m%d %H%M%S') for x in inds_.index.values]

            
        #     d = item[1].reset_index(('date','code'),drop=True)

            ax.set_title(item[0],color='r', loc ='left', pad=-10) 
            DD = inds_[['DIFF','DEA']]
            DD.index = index_
            DD.plot(kind='line', ax=ax)
            macd = inds_['MACD']
            macd.index = index_
            macd.plot(kind='bar', ax=ax)
        #     print([pd.to_datetime(x) for x in inds_.index.values])
        #     print(inds_.index)
        #     print(inds_['MACD'].index)
        #     print(inds_[['DIFF','DEA']].index)

        #     print(inds_['MACD'])
        #     print(inds_[['DIFF','DEA']])
    