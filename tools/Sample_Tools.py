
import os 
import sys 
module_path = os.path.abspath(os.path.join('..')) 
if module_path not in sys.path: 
    sys.path.append(module_path)

import QUANTAXIS as QA
from QUANTAXIS.QAData.financial_mean import financial_dict
from QUANTAXIS.QAUtil.QAParameter import MARKET_TYPE, RUNNING_ENVIRONMENT, ORDER_DIRECTION
from QUANTAXIS.QAUtil import  trade_date_sse

from QUANTAXIS.QAData.data_marketvalue import QA_data_marketvalue
from QUANTAXIS.QAUtil import (
        DATABASE,
        QA_util_code_tolist,
        QA_util_date_stamp,
        QA_util_date_valid,
        QA_util_log_info,
)

from QUANTAXIS.QAUtil.QADate_trade import (
    QA_util_get_pre_trade_date,
    QA_util_get_next_trade_date,
    QA_util_if_tradetime
)

import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy.stats as st
import numpy as np
import pandas as pd

import time
import datetime
import re

from base.JuUnits import excute_for_multidates
import base.JuUnits as u

mpl.rcParams['font.sans-serif'] = ['SimHei','KaiTi', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
mpl.rcParams['font.size'] = 14  # 字体大小
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号

from sklearn.neighbors import KernelDensity  




########### data #################
def get_data(codes_list, start=None, end=None, gap=60, freq=QA.FREQUENCE.DAY, market=MARKET_TYPE.STOCK_CN):
    ''':param freq： --只能比day频率高，需要更低频需要从day重新采样。
    '''
    assert ((not start is None) or (not end is None)), 'start 和 end 必须有一个'
    if start is None:
        start_ = QA_util_get_next_trade_date(end, gap*-1)  # trade_date_sse[trade_date_sse.index(end) - gap]
    else:
        start_ = start
    if end is None:
        end_ = QA_util_get_next_trade_date(start, gap)# trade_date_sse[trade_date_sse.index(start) + gap]
    else:
        end_ = end
        
    if freq == QA.FREQUENCE.DAY:
        data = QA.QA_fetch_stock_day_adv(codes_list, start_, end_)
    else:
        data = QA.QA_fetch_stock_min_adv(codes_list, start_, end_,frequence=freq)

            
#     data = QA.QA_quotation(codes_list, start_, end_, source=QA.DATASOURCE.MONGO,
#                                frequence=freq, market=market, 
#                                output=QA.OUTPUT_FORMAT.DATASTRUCT)
    return data

def get_index_data(code, start=None, end=None, gap=60, freq=QA.FREQUENCE.DAY):
    ''':param freq： --只能比day频率高，需要更低频需要从day重新采样。
    '''
    assert ((not start is None) or (not end is None)), 'start 和 end 必须有一个'
    if start is None:
        start_ = QA_util_get_next_trade_date(end, gap*-1)(end, gap*-1)  # trade_date_sse[trade_date_sse.index(end) - gap]
    else:
        start_ = start
    if end is None:
        end_ = QA_util_get_next_trade_date(start, gap)# trade_date_sse[trade_date_sse.index(start) + gap]
    else:
        end_ = end
        
    if freq == QA.FREQUENCE.DAY:
        data = QA.QA_fetch_index_day_adv(code, start_, end_)
    else:
        data = QA.QA_fetch_index_min_adv(code, start_, end_,frequence=freq)

    return data

def resample_stockdata_low(stock_df,freq="M"):
    '''低频数据重新按时间降采样, 月周天采样
        :param freq：{str} --in[?D，?w, ?M, Q]
        注：day采样取左, week,month取右
    '''
    if np.sum([char in ["d",'D'] for char in list(freq)])>0:
        tmp = stock_df.groupby(level=1).apply(lambda x: x.reset_index('code',drop=True).resample(freq,closed='left', label='left').first())
        tmp = tmp.dropna()
    else:
        tmp = stock_df.groupby(level=1).apply(lambda x: x.reset_index('code',drop=True).resample(freq,closed='right', label='right').last())
    #groupby level=1，导致index顺序翻转，转回原index顺序
    return tmp.reset_index().set_index(['date','code'])

def resample_stockdata_high(stock_df,freq="5min"):
    '''高频数据重新按时间降采样 
    '''
    pass

def get_stock_name(code):
    if isinstance(code, list):
        return QA.QA_fetch_stock_name(code).name
    if isinstance(code, str):
        return QA.QA_fetch_stock_name(code)
    raise TypeError('code MUST BE list or str')

    
########### samples #################
def get_codes_by_zs(name='沪深300', only_main=True,filter_st=True):
    codes_list = get_codes_by_market(codes_list=get_blocks_view('zs')[name], sse='all',only_main=only_main,filter_st=filter_st)
    return codes_list

def get_sample_by_zs(name='沪深300', start=None, end=None, gap=60, freq=QA.FREQUENCE.DAY, only_main=True, filter_st=True):
    codes_list = get_codes_by_zs(name, only_main=only_main,filter_st=True)
    data = get_data(codes_list, start=start, end=end, gap=gap, freq=freq, market=MARKET_TYPE.STOCK_CN)
    return data


  
########### benchmark samples #################
def get_benchmark(name=None, code=None, start=None, end=None, gap=60, freq=QA.FREQUENCE.DAY):
    '''优先name，如果name不为空，但未查询到，会报错处理
    '''
    assert ((not name is None) or (not code is None)), 'name 和 code 必须有一个'
    assert ((not start is None) or (not end is None)), 'start 和 end 必须有一个'
    
    code_ = code
    if name is not None:
        index_list = QA.QA_fetch_index_list_adv()
        target = index_list[index_list['name']==name]
        assert len(target) > 0 , "name: %s 未查询到" % name
        code_ = target.code.tolist()
        if name == '沪深300':
            # 沪深300会查出上证和深证的两个代码，
            code_ = code_[0:1]
    
    return get_index_data(code_, start=start, end=end, gap=gap, freq=freq)

    
########### indicator #################
def get_forward_return(stocks_df,column):
    '''计算(未来)下一个回报率
    :param stocks_df: {pd.DataFrame 或 stock_struct}
    :param column: {string}
    :return: {pd.Series}
    '''
    ret = stocks_df[column].groupby(level=1, group_keys=False).apply(lambda x:(x/x.shift(1)-1).shift(-1))
    ret.name = 'ret_forward'
    return ret

def get_current_return(stocks_df,column,stride=1):
    '''计算当期的回报率
    :param stocks_df: {pd.DataFrame 或 stock_struct}
    :param column: {str} --用于计算收益的列名
    :param stride: {int} --计算收益的跨度
    注意：当期回报有可能也包含未来信息。
    '''
    ret = stocks_df[column].groupby(level=1, group_keys=False).apply(lambda x:(x/x.shift(stride)-1))
    ret.name = 'ret'
    return ret

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

def add_industry(stocks_df, hy_source='swhy', inplace=True):
    '''向stock的DataFrame中插入行业数据
        :param stocks_df: --stock的DataFrame
        :param hy_source: {str in ['gn', 'tdxhy', 'zs', 'fg', 'swhy' 'yb', 'csindex']} --指明数据来源
        :param inplace:{bool} --是否在原对象上修改,建议用true，效率高（default:True）
       :return: {pd.DataFame}
    '''
    industry = get_blockname_from_stock(stocks_df.index.levels[1].to_list(), hy_source)
    if inplace:
        stocks_df['industry'] = stocks_df.index.get_level_values(1).map(industry)
        return stocks_df
    else:
        return stocks_df.assign(industry=stocks_df.index.get_level_values(1).map(industry))

def add_report_inds(data_df, inds_names=['totalCapital']):
    codes = data_df.index.get_level_values(1).unique().tolist()
    date_ = data_df.index.get_level_values(0)
    date_start = get_pre_report_date(date_.min())
    date_end = get_next_report_date(date_.max())
    report_df = QA.QA_fetch_financial_report_adv(codes, date_start,date_end,ltype='EN').data[inds_names]

    
    data_df['report_date'] = data_df.apply(lambda x:pd.Timestamp(get_pre_report_date(x.name[0])) ,axis=1)
    data_re =  data_df.reset_index().set_index(['report_date','code'])
    
    data_merge = pd.merge(data_re,report_df, left_index=True, right_index=True, how='inner')
    
    data_merge_re = data_merge.reset_index().set_index(['date','code'])
    del data_merge_re['report_date']
    
    return data_merge_re

def add_marketvalue_industry(df:pd.DataFrame, static_mv:bool=False):
    '''市值，行业-中性化：
        :param df：{pd.DataFrame} --需要中性化的指标
        :param static_mv：{bool} --是否使用静态市值，静态市值取自财报， 动态市值通过复权信息和收盘价进行计算。
        注意：动态市值区分总股本和流动股本。totalCapital 和 liquidity_totalCapital
    '''  
    df_ = df
    if not isinstance(df, pd.DataFrame):
        df_ = pd.DataFrame(df)
        
    # 静态市值取自财报， 动态市值通过复权信息和收盘价进行计算。
    if static_mv:
        df_reported = add_report_inds(df_, inds_names=['totalCapital'])
    else:
        df_reported = QA_data_marketvalue(df_)
        df_reported.rename(columns=lambda x:x.replace('mv','totalCapital'), inplace=True)
        
    add_industry(df_reported)
    
    df_reported.dropna(axis=0,inplace=True)
    df_reported.drop(df_reported[df_reported['totalCapital']==0].index,inplace=True)

    return df_reported

########### block #################
def get_all_blocks(hy_source='swhy', collections=DATABASE.stock_block):
    a = QA.QA_fetch_stock_block_adv(collections=collections).data
    return a[a['type']==hy_source].index.get_level_values('blockname').unique().to_list()

def get_blocks_view(hy_source, collections=DATABASE.stock_block):
    ''' 获取行业视图，既行业以及对应的code
        :param hy_source: {str in ['gn', 'tdxhy', 'zs', 'fg', 'swhy' 'yb', 'csindex', 'concept', 'industry']} --指明数据来源
        其中'concept', 'industry' 在表DATABASE.stock_block_em中
    
    '''
    if not hy_source in ['gn', 'tdxhy', 'zs', 'fg', 'swhy', 'yb', 'csindex', 'concept', 'industry']:
        raise TypeError('hy_source MUST BE [gn|tdxhy|zs|fg|swhy|yb|csindex|concept|industry]')

#     a = QA.QA_fetch_stock_block_adv().data
#     blocks_view = a[a['type'] == hy_source].groupby(level=0).apply(
#         lambda x:[item for item in x.index.remove_unused_levels().levels[1]]
#     )
    a = QA.QA_fetch_stock_block(collections=collections)
    blocks_view = a[a['type'] == hy_source].groupby('blockname').apply(lambda x:x.index.to_list())
    return blocks_view

def get_blockname_from_stock(code, hy_source='swhy', collections=DATABASE.stock_block):
    code_ = code
    if isinstance(code, str):
        code_ = [code]
    try:
        res = QA.QA_fetch_stock_block(code=code_, collections=collections)
    except Exception as e:
        print(e,'get_stock_blockname ：code error')
        return None
    return res[res['type']==hy_source]['blockname']

def get_codes_from_blockname(blockname, sse='all', only_main=True, filter_st=True, collections=DATABASE.stock_block):
    codes = QA.QA_fetch_stock_block_adv(blockname=blockname, collections=collections).code
    if len(codes)!=0:
        codes = get_codes_by_market(codes_list=codes, sse=sse, only_main=only_main,filter_st=filter_st)
    return codes

def get_codes_by_market(codes_list=None, sse='sh',only_main=True,filter_st=True):
    '''按照市场获取股票代码，分为3类sh，sz，all，默认只获取主板；
       :param codes_list: --为空时，从全市场codes中获得整个子市场的codes;
                            不为空时，则过滤codes_list的内容;
                          (default: None)
       :param sse:{str in ['all', 'sh', 'sz']} --市场名称
       :param only_main:{bool} --是否主板，True时，过滤创业板内容。(default: True)
       :param filter_st:{bool} --是否过滤ST，仅在codes_list为空的时候生效。(default: True)
    '''
    condition = []
    if sse =='all':
        condition.append('60')
        condition.append('00')
        if not only_main:
            condition.append('30')
            condition.append('68')
    if sse == 'sh':
        condition.append('60')
        if not only_main:
            condition.append('68')
    if sse == 'sz':
        condition.append('00')
        if not only_main:
            condition.append('30')
            
    assert len(condition), '参数错误，检查sse内容'
    
    stocks = QA.QA_fetch_stock_list()
    if codes_list is None:
        if filter_st:
            stocks = stocks[~stocks.name.str.startswith('ST')]
        return stocks[stocks.code.map(lambda x:x[0:2] in condition)].code.unique().tolist()
    else:
        stocks = stocks[stocks.index.isin(codes_list)]
        if filter_st:
            stocks = stocks[~stocks.name.str.startswith('ST')]
        return stocks[stocks.code.map(lambda x:x[0:2] in condition)].code.unique().tolist()

    
    
###############  financial  #########################
def get_quarter_list(start_year, end_year, quarter_ordesr=[1,2,3,4], generate_label=False):
    '''生成财报季报的发布日期，
       :param start_year:{int|str} --起始年份;
       :param end_year:{int|str} --结束年份;
       :param quarter_ordesr:{list[int] in [1, 2, 3, 4]} --季度标识，标识需要产生哪个季度的日期;         
       :param generate_label:{bool} --是否生成季度标签。(default: False)
       :return：{list,tuple[list]} -- 默认返回季报发布日期，
                                      如果generate_label=True时返回tuple(date,label)，
    '''
    if len(set(quarter_ordesr).union([1,2,3,4])) > 4:
        raise TypeError('quarter_ordesr MUST IN [1|2|3|4]')
        
    date_element = ['-03-31', '-06-30', '-09-30', '-12-31']
    lables = ['Q1', 'Q2', 'Q3', 'Q4']
    
    res_date = []
    res_lables = []
    for order in np.array(quarter_ordesr)-1:
        res_date.extend([str(y)+date_element[order] for y in range(int(start_year), int(end_year)+1)])
        if generate_label:
            res_lables.extend([str(y)+lables[order] for y in range(int(start_year), int(end_year)+1)])
            
    res_date.sort()
    if generate_label:
        res_lables.sort()
        return res_date,res_lables
    return res_date

def get_next_report_date(cur_date):
    cur_date_ = cur_date
    if isinstance(cur_date, pd.Timestamp):
        cur_date_ = cur_date.strftime('%Y-%m-%d')
        
    assert len(re.findall(r"(\d{4}-\d{2}-\d{2})",cur_date_)) > 0, '日期格式必须为：xxxx-xx-xx；当前输入：%s' % cur_date_
    
    year = cur_date_[0:4]
    month_day = cur_date_[4:10]
    date_element = ['-03-31', '-06-30', '-09-30', '-12-31']
    if month_day in date_element:
        index = date_element.index(month_day)
        next_index = index +1 if index != 3 else 0
        year = str(int(year)+1) if next_index==0 else year
        return year+date_element[next_index]
    else:
        month = cur_date_[5:7]
        date_element = ['-03-31','-03-31','-03-31','-03-31','-06-30','-06-30','-06-30','-09-30','-09-30','-09-30','-12-31','-12-31','-12-31']
        return year+date_element[int(month)]

def get_pre_report_date(cur_date, keep_current=True):
    cur_date_ = cur_date
    if isinstance(cur_date, pd.Timestamp):
        cur_date_ = cur_date.strftime('%Y-%m-%d')
        
    assert len(re.findall(r"(\d{4}-\d{2}-\d{2})",cur_date_)) > 0, '日期格式必须为：xxxx-xx-xx；当前输入：%s' % cur_date_
    
    year = cur_date_[0:4]
    month_day = cur_date_[4:10]
    date_element = ['-03-31', '-06-30', '-09-30', '-12-31']
    if month_day in date_element:
        if keep_current:
            return cur_date # 当前正好是report_date的返回。
        index = date_element.index(month_day)
        pre_index = index-1 if index >=0 else 3
        year = str(int(year)-1) if pre_index==3 else year
        return year+date_element[pre_index]
    else:
        month = cur_date_[5:7]
        date_element = ['-12-31','-12-31','-12-31','-12-31','-03-31','-03-31','-03-31','-06-30','-06-30','-06-30','-09-30','-09-30','-09-30']
        if int(month) <=3:
            return str(int(year)-1)+date_element[int(month)]
        return year+date_element[int(month)]
    
###############  other  #########################


