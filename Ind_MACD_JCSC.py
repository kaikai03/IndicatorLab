
import numpy as np
import pandas as pd
import copy

import QUANTAXIS as QA
from QAIndicatorStructExt import QA_DataStruct_Indicators_Ext

class Ind_Model:
    def __init__(self,data):
#         if not isinstance(data, type(QA.OUTPUT_FORMAT.DATASTRUCT)):
#             raise TypeError('Must be DATASTRUCT')
        self.data = data
        self.ind_df = None
        self.decision_df = None
        

    @property
    def ind(self):
        return QA_DataStruct_Indicators_Ext(self.ind_df)

    @property
    def decision(self):
        return QA_DataStruct_Indicators_Ext(self.decision_df)
     
    def on_indicator_structure(self, data):
        raise NotImplementedError
        
    def on_desition_structure(self, data, ind_data):
        raise NotImplementedError

    def fit(self):
        self.ind_df = self.on_indicator_structure(self.data)
        try:
            self.decision_df = self.on_desition_structure(self.data, self.ind_df)
        except NotImplementedError:
            self.decision_df = None
        except Exception as err:
            raise Exception(err)
            

class MACD(Ind_Model):
    def __init__(self,data):
        super().__init__(data)
        self._pramas_default = {}
        self.pramas = copy.deepcopy(self._pramas_default)
        
    def on_indicator_structure(self, data):
        return self.MACD_JCSC(data)
        
    def MACD_JCSC(self,dataframe,SHORT=12,LONG=26,M=9):
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

    