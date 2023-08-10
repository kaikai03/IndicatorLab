
import sys
import os 
module_path = os.path.abspath(os.path.join('..')) 
if module_path not in sys.path: 
    sys.path.append(module_path)
    
import numpy as np
import pandas as pd

import QUANTAXIS as QA

import Ind_Model_Base

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tools.Sample_Tools as smpl
import tools.Pretreat_Tools as pretreat
import Analysis_Funs as af

# %load_ext autoreload
# %autoreload 1
# %aimport Pretreat_Tools,Sample_Tools


class PondingRate(Ind_Model_Base.Ind_Model):
    """对数积水率
        [-1.1]，实际面积与回撤(回调)面积的比（相当于差的积分）
        有方向的，1相当于正向，无齿波，越接近0，抖动频率和幅度越大。
        月现，日线大体上表现一致。
        主要指标是稳健积水率，先对数据做个回归，用回归线再作为实际面积的计算。
        次要指标是使用原始的实际面积
        回报与指标反向。
    """
    optimum_param={'valid':True, 'main':'stably', 'desition_direct':-1, 'window':14, 'freq':'d','neutralize':{'enable':True,'static_mv':False}}
    def __init__(self,data, frequence=QA.FREQUENCE.DAY):
        super().__init__(data, 'PondingRate', frequence)


    def on_set_params_default(self):
        return {'window':14,'moving':5}
    
        
    def on_indicator_structuring(self, data):
        return self.excute_for_multicode(data, self.kernel, **self.pramas)

    
#     def on_desition_structuring(self, data, ind_data):
#         """
#         """
    def getPastLogHigh(self, logVs):
        logHs = [logVs[0]]
        for logV in logVs[1:]:
            if logV>logHs[-1]:
                logHs.append(logV)
            else:
                logHs.append(logHs[-1])
        logHs = np.array(logHs)
        return logHs
    
    # 对数积水率
    def getPoolRate(self, logVs):
        logHs = self.getPastLogHigh(logVs)
        gain = np.sum(logVs) - logVs[0] * len(logVs)
        lost_and_gain = np.sum(logHs) - logHs[0] * len(logHs)
        ninfp1_to_n1p1 = lambda x: 2 / (2 - x) - 1
        poolRate = None
        if gain == 0:
            poolRate = 0
        elif lost_and_gain == 0:
            poolRate = float('-inf')
        else:
            poolRate = gain / lost_and_gain
        poolRate = ninfp1_to_n1p1(poolRate)
        return poolRate
    
    # 稳健对数积水率
    def getPoolRate_stably(self, logVs):
        logHs = self.getPastLogHigh(logVs)
        gain = np.sum(logVs) - logVs[0] * len(logVs)
        lost_and_gain = np.sum(logHs) - logHs[0] * len(logHs)

        lost = lost_and_gain - gain

        dts = np.arange(len(logVs))
        params = af.get_LR_params_fast(dts,logVs)
        s = params[0]
        m = params[-1]
        predLogVs = s * dts + m
        gain = (predLogVs[-1] - predLogVs[0]) * len(logVs) / 2
        lost_and_gain = np.abs(lost + gain)
        ninfp1_to_n1p1 = lambda x: 2 / (2 - x) - 1

        poolRate = None
        if gain == 0:
            poolRate = 0
        elif lost_and_gain == 0:
            poolRate = float('-inf')
        else:
            poolRate = gain / lost_and_gain

        poolRate = ninfp1_to_n1p1(poolRate)
        return poolRate
        
    def kernel(self,dataframe, window=14, moving=5):
        CLOSE = dataframe.close
        CLOSE_log = np.log(CLOSE)
        ind_stably = CLOSE_log.rolling(window).apply(lambda x:self.getPoolRate_stably(x))
        
        if self.ignore_sub_and_desition:
            return pd.DataFrame({'stably':ind_stably}) 
            
        ind_unstably = CLOSE_log.rolling(window).apply(lambda x:self.getPoolRate(x))
        return pd.DataFrame({'stably':ind_stably, 'unstably':ind_unstably})

    
    def plot(self,):
        groups = self.ind_df.groupby(level=1)
        fig = plt.figure(figsize=(1120/72,210*len(groups)/72))
        for idx,item in enumerate(groups):
            inds_ = item[1].reset_index('code',drop=True)
            ax = fig.add_subplot(len(groups),1,idx+1)
            
            
            ##axis不转成字符串的话，bar和line的x轴有时候对不上，原因未知
            formater = '%Y%m%d' if self.is_low_frequence else '%Y%m%d %H%M%S'
            index_ = [pd.to_datetime(x).strftime(formater) for x in inds_.index.values]
#             d = item[1].reset_index(('date','code'),drop=True)

            ax.set_title(item[0],color='blue', loc ='left', pad=-10) 
    
            close = self.data.close.loc[(slice(None),item[0])]
            close.index = index_
            close.plot(kind='line', ax=ax)
            
            ax2 = ax.twinx()
            ax2.set_ylim([-1,1])
            
            main = inds_[PondingRate.optimum_param['main']]
            main.index = index_
            main.plot(kind='line', color='black', ax=ax2,label='stably')
            
            if not self.ignore_sub_and_desition:
                sub = inds_['unstably']
                sub.index = index_
                sub.plot(kind='line', ax=ax2, color='grey',label='unStably')
                ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
                
            plt.legend(loc='lower left', fontsize=10) 
            plt.xticks(rotation = 0)
    

    
