
import QUANTAXIS as QA
from QUANTAXIS.QAData.financial_mean import financial_dict
from QUANTAXIS.QAUtil.QAParameter import MARKET_TYPE, RUNNING_ENVIRONMENT, ORDER_DIRECTION
from QUANTAXIS.QAUtil import  trade_date_sse

import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy.stats as st
import numpy as np
import pandas as pd

import sweetviz as sv

import time
import datetime
import re

from base.JuUnits import excute_for_multidates
import base.JuUnits as u

mpl.rcParams['font.sans-serif'] = ['SimHei','KaiTi', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
mpl.rcParams['font.size'] = 14  # 字体大小
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号

from sklearn.neighbors.kde import KernelDensity  


########### data #################
def get_data(codes_list, start=None, end=None, gap=60, freq=QA.FREQUENCE.DAY, market=MARKET_TYPE.STOCK_CN):
    ''':param freq： --只能比day频率高，需要更低频需要从day重新采样。
    '''
    assert ((not start is None) or (not end is None)), 'start 和 end 必须有一个'
    if start is None:
        start_ = trade_date_sse[trade_date_sse.index(end) - gap]
    else:
        start_ = start
    if end is None:
        end_ = trade_date_sse[trade_date_sse.index(start) + gap]
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
        start_ = trade_date_sse[trade_date_sse.index(end) - gap]
    else:
        start_ = start
    if end is None:
        end_ = trade_date_sse[trade_date_sse.index(start) + gap]
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
def get_sample_by_zs(name='沪深300', start=None, end=None, gap=60, freq=QA.FREQUENCE.DAY, only_main=True):
    codes_list = get_codes_by_market(codes_list=get_blocks_view('zs')[name], sse='all',only_main=only_main)
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
    ret = stocks_df[column].groupby(level=1, group_keys=False).apply(lambda x:((x-x.shift(1))/x.shift(1)).shift(-1))
    ret.name = 'ret_forward'
    return ret

def get_current_return(stocks_df,column):
    '''计算当期的回报率
    :param stocks_df: {pd.DataFrame 或 stock_struct}
    注意：当期回报有可能也包含未来信息。
    '''
    return stocks_df[column].groupby(level=1, group_keys=False).apply(lambda x:(x-x.shift(1))/x.shift(1))

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

    


########### block #################
def get_all_blocks(hy_source='swhy'):
    a = QA.QA_fetch_stock_block_adv().data
    return a[a['type']==hy_source].index.get_level_values('blockname').unique().to_list()

def get_blocks_view(hy_source):
    ''' 获取行业视图，既行业以及对应的code
        :param hy_source: {str in ['gn', 'tdxhy', 'zs', 'fg', 'swhy' 'yb', 'csindex']} --指明数据来源
    
    '''
    if not hy_source in ['gn', 'tdxhy', 'zs', 'fg', 'swhy', 'yb', 'csindex']:
        raise TypeError('hy_source MUST BE [gn|tdxhy|zs|fg|swhy|yb|csindex]')

#     a = QA.QA_fetch_stock_block_adv().data
#     blocks_view = a[a['type'] == hy_source].groupby(level=0).apply(
#         lambda x:[item for item in x.index.remove_unused_levels().levels[1]]
#     )
    a = QA.QA_fetch_stock_block()
    blocks_view = a[a['type'] == hy_source].groupby('blockname').apply(lambda x:x.index.to_list())
    return blocks_view

def get_blockname_from_stock(code, hy_source='swhy'):
    code_ = code
    if isinstance(code, str):
        code_ = [code]
    try:
        res = QA.QA_fetch_stock_block(code=code_)
    except Exception as e:
        print(e,'get_stock_blockname ：code error')
        return None
    return res[res['type']==hy_source]['blockname']

def get_codes_from_blockname(blockname, sse='all', only_main=True):
    codes = QA.QA_fetch_stock_block_adv(blockname=blockname).code
    if sse != 'all' and len(codes)!=0:
        codes = get_codes_by_market(sse=sse, only_main=only_main, codes_list=codes)
    return codes

def get_codes_by_market(codes_list=None, sse='sh',only_main=True):
    '''按照市场获取股票代码，分为3类sh，sz，all，默认只获取主板；
       :param codes_list: --为空时，从全市场codes中获得整个子市场的codes;
                            不为空时，则过滤codes_list的内容;
                          (default: None)
       :param sse:{str in ['all', 'sh', 'sz']} --市场名称
       :param only_main:{bool} --是否主板，True时，过滤创业板内容。(default: True)
      
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
    assert len(re.findall(r"(\d{4}-\d{2}-\d{2})",cur_date)) > 0, '日期格式必须为：xxxx-xx-xx；当前输入：%s' % cur_date
    year = cur_date[0:4]
    month_day = cur_date[4:10]
    date_element = ['-03-31', '-06-30', '-09-30', '-12-31']
    if month_day in date_element:
        index = date_element.index(month_day)
        next_index = index +1 if index != 3 else 0
        year = str(int(year)+1) if next_index==0 else year
        return year+date_element[next_index]
    else:
        month = cur_date[5:7]
        date_element = ['-03-31','-03-31','-03-31','-03-31','-06-30','-06-30','-06-30','-09-30','-09-30','-09-30','-12-31','-12-31','-12-31']
        return year+date_element[int(month)]


    
###############  other  #########################

