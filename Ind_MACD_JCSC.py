
import numpy as np
import pandas as pd
import copy

import QUANTAXIS as QA
from QAIndicatorStructExt import QA_DataStruct_Indicators_Ext

class Ind_Model:
    def __init__(self,data, pramas_default=None):
#         if not isinstance(data, type(QA.OUTPUT_FORMAT.DATASTRUCT)):
#             raise TypeError('Must be DATASTRUCT')
        if pramas_default is None:
            self._pramas_default = self.on_set_params_default()
        else:
            self._pramas_default = pramas_default
            
        if not isinstance(self._pramas_default, dict):
            raise TypeError('_pramas_default MUST BE DICT')
            
        self.pramas = copy.deepcopy(self._pramas_default)
        self.data = data
        self.ind_df = None
        self.decision_df = None
        
    def change_pramas(self,**dic):
        for k in dic.keys():
            self.pramas[k] = dic[k]
    
    def reset_pramas(self):
        self.pramas = copy.deepcopy(self._pramas_default)
    
    @property
    def cur_pramas(self):
        return self.pramas
    @property
    def default_pramas(self):
        return self._pramas_default
    @property
    def keys_pramas(self):
        return self._pramas_default.keys()    

    @property
    def ind(self):
        return QA_DataStruct_Indicators_Ext(self.ind_df)

    @property
    def decision(self):
        if self.decision_df is None  or not isinstance(self.decision_df, pd.DataFrame):
            print("decision_df:",self.decision_df)
            raise Exception('on_desition_structure error')
        return QA_DataStruct_Indicators_Ext(self.decision_df)
     
        
    def on_set_params_default(self) -> dict:
        raise NotImplementedError
        
    def on_indicator_structure(self, data):
        raise NotImplementedError
        
    def on_desition_structure(self, data, ind_data) -> pd.DataFrame:
        raise NotImplementedError

    def fit(self):
        self.ind_df = self.on_indicator_structure(self.data)
        try:
            self.decision_df = self.on_desition_structure(self.data, self.ind_df)
        except NotImplementedError:
            self.decision_df = None
        except Exception as err:
            raise Exception(err)
            

class MACD_JCSC(Ind_Model):
    def __init__(self,data):
        super().__init__(data)

    def __repr__(self):
        return '< MACD in pramas  {}  >'.format(str (self.pramas))
    
    def on_set_params_default(self):
        return {'SHORT':12,'LONG':26,'M':9}
        
    def on_indicator_structure(self, data):
        return self.MACD_JCSC(data,**self.pramas)
    
    def on_desition_structure(self, data, ind_data):
        return None
        
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

    