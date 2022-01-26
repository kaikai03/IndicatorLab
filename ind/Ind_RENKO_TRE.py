
import sys
import os 
module_path = os.path.abspath(os.path.join('..')) 
if module_path not in sys.path: 
    sys.path.append(module_path)

import Ind_Model_Base

import numpy as np
import pandas as pd
import math

import QUANTAXIS as QA


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import patches

import numba as nb
import talib
import scipy.optimize as opt

import Analysis_Funs as af

# %load_ext autoreload
# %autoreload 2
# %aimport Analysis_Funs,Ind_Model_Base


class RENKO(Ind_Model_Base.Ind_Model):
    optimum_param={'valid':True, 'main':'feature_RENKO_JXSX', 'desition_direct':1, 'freq':'d','neutralize':{'enable':True,'static_mv':False}}
    
    def __init__(self,data, frequence=QA.FREQUENCE.DAY, sensitive_mode=False):
        super().__init__(data, 'RENKO', frequence)
        self.sensitive_mode = sensitive_mode
        self.renko_objs={}
        
        ### 正式使用时由实例设置为true来加速
#         self.set_ignore_sub_ind(False)
        

    def on_set_params_default(self):
        return {'timeperiod':14}
    
        
    def on_indicator_structuring(self, data):
        return self.excute_for_multicode(data, self.kernel, **self.pramas)
    
    def on_desition_structuring(self, data, ind_data):
        """
        """
        JXSX = self.excute_for_multicode(self.ind_df,
                                         lambda x: pd.DataFrame(af.feature_JXSX_timeline(x['direct']),
                                         index=x.index,
                                         columns=['feature_RENKO_JXSX']))
        continuity = self.excute_for_multicode(self.ind_df, 
                                               lambda x: pd.DataFrame(af.timeline_event_continuity(x['direct'].values),
                                               index=x.index,
                                               columns=['feature_RENKO_CONTINUITY']))
#         continuity['feature_RENKO_CONTINUITY_ABS'] = np.abs(continuity['feature_RENKO_CONTINUITY'])
        
        self.ind_df = pd.concat([self.ind_df, JXSX, continuity],axis=1)
        return None #pd.concat([JXSX,continuity],axis=1)
        
    def kernel(self,dataframe,timeperiod=14):
        hlc = dataframe[['high','low','close']]
        if len(hlc)<2:
            return None
        
        optimal_brick = get_optimal_brick_ML(hlc, self.cur_pramas['timeperiod'])
        if optimal_brick is None:
            return None
        
        if self.sensitive_mode:
            renko_objc = renko_chart_sensitive(hlc.close.values, optimal_brick,condensed=True)
            ret_indices = pd.DataFrame(renko_objc[:,0:3], 
                                       columns=['low_band', 'upper_band', 'direct'], 
                                       index=hlc.index)
        else:
            renko_objc = renko_chart()
            renko_objc.set_brick_size(brick_size = optimal_brick, auto = False)
            renko_objc.build_history(prices = hlc.close.values)
            # up时，boundary为上边界，down时，boundary为下边界
            ret_indices = pd.DataFrame(renko_objc.source_aligned, 
                                       columns=['boundary', 'initial', 'direct'], 
                                       index=hlc.index)
            
        if not self.fast_mode:
            self.renko_objs[dataframe.index.get_level_values(1)[0]] = renko_objc
        
        
        
        if self.fast_mode:
            return ret_indices.iloc[:,2:3]


        return ret_indices

    def plot(self):
        if self.fast_mode:
            raise 'fast_mode is True,cant draw chart'
        if self.sensitive_mode:
            for i in self.renko_objs.keys():
                plot_renko_chart(self.renko_objs[i],i)
        else:
            for i in self.renko_objs.keys():
                self.renko_objs[i].plot_renko(chart_name=i)
        


