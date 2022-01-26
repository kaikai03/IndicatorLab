
import os 
import sys 
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

from base.JuUnits import roll_multi_result

%load_ext autoreload
%autoreload 2
%aimport tools.Pretreat_Tools,tools.Sample_Tools,base.JuUnits


class RetraceRate(Ind_Model_Base.Ind_Model):
    optimum_param={'valid':True, 'main':'max_retrace_rate', 'desition_direct':-1, 'window':14, 'freq':'d','neutralize':{'enable':False,'static_mv':False}}
    """回撤（回吐）类指标
        main 默认是回撤率，
        sub（含main）顺序为：回撤率,最大回撤周期,最大回撤,最大下跌恢复周期，最大过山车周期
    """
    
    def __init__(self,data, frequence=QA.FREQUENCE.DAY):
        super().__init__(data, 'RetraceRate', frequence)
    
    def on_set_params_default(self):
        return {'window':14,'moving':5}
    
        
    def on_indicator_structuring(self, data):
        return self.excute_for_multicode(data, self.kernel, **self.pramas)

    
#     def on_desition_structuring(self, data, ind_data):
#         """
#         """

        
    def kernel(self,dataframe, window=14, moving=5):
        CLOSE = dataframe.close
        if self.ignore_sub_and_desition:
            retraces = CLOSE.rolling(window).apply(lambda x:af.retracing(x)[1])
            return pd.DataFrame({'main':retraces}) 
        else:
            def dealing(arr):
                data_ = arr[:,2]
#                 print(data_)
                retrace = af.retracing(data_)
                climb = af.climbing(data_)
                recover = af.get_longest_recover_duration(data_,af.cross_sign(af.get_direct_sign(data_)))[0]
                rollercoaster = af.get_longest_rollercoaster_duration(data_,af.cross_sign(af.get_direct_sign(data_)))[0]
                
                # 回撤率,最大回撤周期,最大回撤,爬升率,最大爬升周期,最大爬升,恢复，过山车
                return retrace[1],retrace[2],retrace[0],climb[1],climb[2],climb[0],recover,rollercoaster
#             
            res = roll_multi_result(CLOSE, dealing, window, 8)
#             print(res)
#             print(CLOSE.index)
            return pd.DataFrame({'max_retrace_rate':res[:,0], 'max_retrace_duration':res[:,1], 'max_retrace':res[:,2], 
                                 'max_climb_rate':res[:,3], 'max_climb_duration':res[:,4], 'max_climb':res[:,5], 
                                 'recovers_duration':res[:,6], 'rollercoaster_duration':res[:,7]},index=CLOSE.index)
  

    
    def plot(self,):
        pass
#         groups = self.ind_df.groupby(level=1)
#         fig = plt.figure(figsize=(1120/72,210*len(groups)/72))
#         for idx,item in enumerate(groups):
#             inds_ = item[1].reset_index('code',drop=True)
#             ax = fig.add_subplot(len(groups),1,idx+1)
            
            
#             ##axis不转成字符串的话，bar和line的x轴有时候对不上，原因未知
#             formater = '%Y%m%d' if self.is_low_frequence else '%Y%m%d %H%M%S'
#             index_ = [pd.to_datetime(x).strftime(formater) for x in inds_.index.values]
# #             d = item[1].reset_index(('date','code'),drop=True)

#             ax.set_title(item[0],color='blue', loc ='left', pad=-10) 
    
#             close = self.data.close.loc[(slice(None),item[0])]
#             close.index = index_
#             close.plot(kind='line', ax=ax)
            
#             ax2 = ax.twinx()
#             ax2.set_ylim([-1,1])
            
#             main = inds_['main']
#             main.index = index_
#             main.plot(kind='line', color='black', ax=ax2,label='stably')
            
#             if not self.ignore_sub_and_desition:
#                 sub = inds_['sub']
#                 sub.index = index_
#                 sub.plot(kind='line', ax=ax2, color='grey',label='unStably')
#                 ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
                
#             plt.legend(loc='lower left', fontsize=10) 
#             plt.xticks(rotation = 0)
    

    
