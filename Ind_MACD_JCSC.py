
import pandas as pd

import QUANTAXIS as QA
from QAIndicatorStructExt import QA_DataStruct_Indicators_Ext

class MACD:
    def __init__(self):
        pass
        
    def MACD_JCSC_data(self,dataframe,SHORT=12,LONG=26,M=9):
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

    def MACD_JCSC(self,dataframe,SHORT=12,LONG=26,M=9):
        return QA_DataStruct_Indicators_Ext(self.MACD_JCSC_data(dataframe,SHORT,LONG,M))