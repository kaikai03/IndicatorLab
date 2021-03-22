
import QUANTAXIS as QA
from QUANTAXIS.QAData.financial_mean import financial_dict
import pandas as pd


import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import sweetviz as sv

import time
import datetime

from base.JuUnits import excute_for_multidates

plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False 

from sklearn.neighbors.kde import KernelDensity  

# class Macro_Descer:
#     def __init__(self, )

def auto_describe(df):
    report = sv.analyze(finances_filted,pairwise_analysis='auto')
    report.show_notebook()
    
def fliter_codes_by_market(sse='sh',only_main=True, codes_list=None):
    '''按照市场获取股票代码，分为3类sh，sz，all，默认只获取主板
       codes_list为空时，从全市场codes中获得整个子市场的codes
    '''
    condition = []
    if sse =='all':
        condition.append('6')
        condition.append('0')
        if not only_main:
            condition.append('3')
            
    if sse == 'sh':
        condition.append('6')
    if sse == 'sz':
        condition.append('0')
        if not only_main:
            condition.append('3')
            
    assert len(condition), '参数错误，检查sse内容'
    if codes_list is None:
        stocks = QA.QA_fetch_stock_list()
        return stocks[stocks.code.map(lambda x:x[0] in condition)].code.unique().tolist()
    else:
        return [code for code in codes_list if code[0] in condition]

def get_Q1_list(start, end):
    return [str(y)+'-03-31' for y in range(int(start), int(end)+1)]

def get_Q2_list(start, end):
    return [str(y)+'-06-30' for y in range(int(start), int(end)+1)]

def get_Q3_list(start, end):
    return [str(y)+'-09-30' for y in range(int(start), int(end)+1)]

def get_Q4_list(start, end):
    return [str(y)+'-12-31' for y in range(int(start), int(end)+1)]


def drop_by_quantile_multidates(obj, floor=.00,upper=1., column=None):
    return excute_for_multidates(obj, drop_by_quantile, floor=floor,upper=upper, column=column).sort_index()

def drop_by_quantile(obj, floor=.00,upper=1., column=None):
    if isinstance(obj, pd.Series):
        qt = obj.quantile([floor,upper])
        return obj[(obj>=qt[floor]) & (obj<=qt[upper])]
    
    if isinstance(obj, pd.DataFrame):
        assert column, 'COLUMN CANT be NONE when obj is dataframe'
        qt = obj[column].quantile([floor,upper])
        return obj[(obj[column]>=qt[floor]) & (obj[column]<=qt[upper])]
        
    raise TypeError('obj must be series or dataframe')

def get_blockname_from_stock(code, type):
    code_ = code
    if isinstance(code, str):
        code_ = [code]
    try:
        res = QA.QA_fetch_stock_block_adv(code=code_)
    except Exception as e:
        print(e,'get_stock_blockname ：code error')
        return None
    res = res.data
    return res[res['type']=='gn'].index.get_level_values('blockname').values

def get_codes_from_blockname(blockname, sse='all', only_main=True):
    codes = QA.QA_fetch_stock_block_adv(blockname=blockname).code
    if sse != 'all' and len(codes)!=0:
        codes = fliter_codes_by_market(sse=sse, only_main=only_main, codes_list=codes)
    return codes

def get_stock_name(code):
    if isinstance(code, list):
        return QA.QA_fetch_stock_name(code).name
    if isinstance(code, str):
        return QA.QA_fetch_stock_name(code)
    raise TypeError('code MUST BE list or str')
    
def get_all_blocks(type_):
    a = QA.QA_fetch_stock_block_adv().data
    return a[a['type']=='type_'].index.get_level_values('blockname').unique().to_list()

def get_rank(data, codes=None,quantile=False, column=None):
    '''get_rank(a,['000001','000002'],column=['totalAssets','ROE'])'''
    if len(data.index.names) >= 2:
        res = excute_for_multidates(data, lambda x: x.rank(ascending=False,pct=quantile))
        if codes:
            if column:
                res=res.loc[pd.IndexSlice[:,codes], column]
            else:
                res=res.loc[pd.IndexSlice[:,codes],slice(None)]
    else:
        res =  data.rank(ascending=False,pct=quantile)
        if codes:
            res = res.loc[codes]
    return res