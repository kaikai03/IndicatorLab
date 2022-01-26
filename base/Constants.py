
import QUANTAXIS as QA
from QUANTAXIS.QAUtil import trade_date_sse
import os

LOW_FREQUENCE = [QA.FREQUENCE.YEAR, QA.FREQUENCE.QUARTER, QA.FREQUENCE.MONTH, QA.FREQUENCE.WEEK, QA.FREQUENCE.DAY]   
PLOT_TITLE = dict(x=0, fontsize='xx-large',fontweight='black',horizontalalignment='left',bbox=dict(boxstyle='roundtooth',fc='lavenderblush', pad=0.25))

CUR_PATH = os.path.abspath(os.path.dirname('.'))
ROOT_PATH = CUR_PATH[:CUR_PATH.find("IndicatorLab")+len("IndicatorLab")] 
BAES_PATH = os.path.abspath(ROOT_PATH + '\\base')

DC_FILEPATH = os.path.join(BAES_PATH, "dcp")
DK_FILEPATH = os.path.join(BAES_PATH, "dkp")
DASK_HOST = '106.14.20.200:7353'
