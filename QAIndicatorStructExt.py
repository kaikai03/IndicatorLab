
import numpy as np
import pandas as pd

from QUANTAXIS.QAData import QA_DataStruct_Indicators
    
class QA_DataStruct_Indicators_Ext(QA_DataStruct_Indicators):
    def __init__(self, data):
#         self.data = data
        super().__init__(data)
        
    def get_ind(self, code=None, date=None):
        ### 白写，和基类的get_timerange相同
        if code and date :
            if isinstance(date,tuple):
                return self.data.loc[pd.IndexSlice[date[0]:date[1],code],:]
            return self.data.loc[pd.IndexSlice[date,code],:]
        if code :
            return self.data.xs(code, level=1)
        if date :
            if isinstance(date,tuple):
                return self.data.loc[date[0]:date[1]] 
            return self.data.loc[date]
            
        raise Exception("不允许两个参数都为空")
        
    def get_ind_offset(self, code, base_date, offset=0):
        if code is None or base_date is None:
            raise Exception("code，date 不允许为空")
        try:
            ind = self.data.xs(code, level=1)
        except:
            return ValueError('CANNOT FOUND THIS CODE')
        
        try:
            idx = ind.index.get_loc(base_date)
        except:
            return ValueError('CANNOT FOUND THIS TIME RANGE')

        if np.sign(offset) < 0 :
            return ind.iloc[idx + offset if idx + offset >=0 else 0:idx+1]
        else:
            return ind.iloc[idx:idx+offset+1]

    def get_ind_for_train(self, code, date):
        '''get当天的就未来函数了'''
        inds = self.get_ind_offset(code, date, offset=-1)
        if len(inds)<=1:
            return [None]
        return inds.iloc[0]
