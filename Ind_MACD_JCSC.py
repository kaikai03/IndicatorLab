
import numpy as np
import pandas as pd

import QUANTAXIS as QA
import Ind_Model_Base


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

    