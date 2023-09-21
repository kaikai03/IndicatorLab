import os 
import sys 
module_path = os.path.abspath(os.path.join('..')) 
if module_path not in sys.path: 
    sys.path.append(module_path)
    
from random import randint

import json
import numpy as np
import pandas as pd

import QUANTAXIS as QA
from QUANTAXIS.QAUtil import (
    DATABASE,
    QA_util_code_tolist,
    QA_util_time_stamp,
    QA_util_date_valid,
    QA_util_date_stamp
)
from QUANTAXIS.QAUtil import (
        QA_util_to_json_from_pandas
)
from QUANTAXIS.QAUtil.QADate_trade import (
        QA_util_get_pre_trade_date,
        QA_util_if_tradetime
)

from QUANTAXIS.QAData import QA_DataStruct_Stock_block


import time
from datetime import (
    datetime as dt,
    timezone, timedelta
)
import pymongo
import traceback
from tqdm.autonotebook import trange, tqdm

from base.JuNetwork import request_json_get
from base.JuUnits import (
    fetch_index_day_common,
    now_time,
    date_range,
)



DATA_KEY={'north':['hk2sh','hk2sz','s2n'],
          'south':['sh2hk','sz2hk','n2s']}

CODE_KEY={'north':'north_',
          'south':'south_'}

TOP10DEAL_TYPE = ['hk2sh','hk2sz','sh2hk','sz2hk']
TOP10DEAL_TYPE2CODE={'hk2sh':'001','hk2sz':'003','sh2hk':'002','sz2hk':'004'}
TOP10DEAL_CODE2TYPE={'001':'hk2sh','003':'hk2sz','002':'sh2hk','004':'sz2hk'}
TOP10DEAL_HEADDIC={'TRADE_DATE':'date','MUTUAL_TYPE':'type',
                   'SECURITY_CODE':'stock_code','DERIVE_SECURITY_CODE':'sse','SECURITY_NAME':'stock_name',
                   'CLOSE_PRICE':'close','CHANGE_RATE':'pct', 'RANK':'rank', 
                   'NET_BUY_AMT':'net_buy_amount','BUY_AMT':'buy_amount','SELL_AMT':'sell_amount','DEAL_AMT':'deal_amount',
                   'DEAL_AMOUNT':'deal_amount_main','MUTUAL_RATIO':'main_ratio'}




def fetch_north_day_from_eastmoney(mode='normal',model='north'):
    '''抓取东方财富--北向日线（基础函数）
        :param mode:{fast|normal|init} --获取数据量的大小，详见内容注释。(default: normal)
        :param model:{north|south} --指定获取南向数据还是北向数据。(default: north)
        注：数值单位是“万”；北向和南向的货币单位不同，对比时注意汇率转换
        http://push2his.eastmoney.com/api/qt/kamt.kline/get?fields1=f1,f3,f5&fields2=f51,f52&klt=101&lmt=500&ut=b2884a393a59ad64002292a3e90d46a5&cb=jQuery112306688839299403427_1633713948773&_=1633713948774
    '''
    assert mode in ['fast','normal','init'], 'mode error,must in {fast|normal|init}'
    assert model in ['north','south'], 'model error,must in {north|south}'
    
    #页面默认一次获取500天数据，最大可获取至2014-11-17沪股通成立，so数据库初始化时可调大，日更时可调小
    lmt = {'fast':30,'normal':500,'init':5000}[mode]
        
    def get_url():
        return "http://push2his.eastmoney.com/api/qt/kamt.kline/get"


    params = {
        'fields1': 'f1,f3,f5' if model=='north' else 'f2,f4,f6',
        'fields2': 'f51,f52',
        'klt': 101,
        'lmt': lmt, 
        'ut': 'b2884a393a59ad64002292a3e90d46a5',
        'cb': 'jQuery112306688839299403427_{:d}'.format(int(dt.utcnow().timestamp())),
        '_': int(time.time() * 1000)
    }
                
    json_data = request_json_get(get_url(), params, mode='jQuery', verbose=False)
    
    
    try:
        if (json_data["data"] is None):
            print('json_data["data"] is None')
            return None
    
        content = json_data["data"]
    except Exception as e:
        print(type(json_data))
        print(json_data)
        traceback.print_exception(type(e), e, sys.exc_info()[2])
        print(u'fetch_north_day_from_eastmoney: failed\n', e) 
        return None

        #print(json_data)
        
        
    # 沪股通 或 港股通（沪）
    hk2sh = pd.DataFrame([item.split(",") for item in content[DATA_KEY[model][0]]],columns=['date','sh_hk']).set_index('date').astype('float64')
    # 深股通 或 港股通（深）
    hk2sz = pd.DataFrame([item.split(",") for item in content[DATA_KEY[model][1]]],columns=['date','sz_hk']).set_index('date').astype('float64')
    # 北向 或 南向
    s2n = pd.DataFrame([item.split(",") for item in content[DATA_KEY[model][2]]],columns=['date','vol']).set_index('date').astype('float64')

    temp_df = pd.concat([hk2sh, hk2sz, s2n],axis=1)
    
    temp_df['code']=CODE_KEY[model]   #兼容QA查询
    temp_df['model']=model
    temp_df['type']= QA.FREQUENCE.DAY
    temp_df["date_stamp"] = pd.to_datetime(temp_df.index).view(np.int64)//10**9     #兼容QA查询
    #print(temp_df)
    return temp_df


def fetch_north_realtime_from_eastmoney(model='north'):
    '''抓取东方财富--北向日线实时
        :param model:{north|south} --指定获取南向数据还是北向数据。(default: north)
        注：数值单位是“万”；北向和南向的货币单位不同，对比时注意汇率转换
        http://push2.eastmoney.com/api/qt/kamt.rtmin/get?fields1=f2,f4&fields2=f51,f52,f54,f56&ut=b2884a393a59ad64002292a3e90d46a5&cb=jQuery112303309361280441192_1633758766690&_=1633758766699
    '''
    assert model in ['north','south'], 'model error,must in {north|south}'

    def get_url():
        return "http://push2.eastmoney.com/api/qt/kamt.rtmin/get"

    params = {
        'fields1': 'f1,f3' if model=='north' else 'f2,f4',
        'fields2': 'f51,f52,f54,f56',
        'ut': 'b2884a393a59ad64002292a3e90d46a5',
        'cb': 'jQuery112308511724156258799_{:d}'.format(int(dt.utcnow().timestamp())),
        '_': int(time.time() * 1000)
    }

    json_data = request_json_get(get_url(), params, mode='jQuery', verbose=False)


    try:
        if (json_data["data"] is None):
            print('json_data["data"] is None')
            return None
    
        content = json_data["data"]
    except Exception as e:
        print(type(json_data))
        print(json_data)
        traceback.print_exception(type(e), e, sys.exc_info()[2])
        print(u'fetch_north_realtime_from_eastmoney: failed\n', e) 
        return None

    # print(json_data)

    DATA_KEY={'north':'s2n','south':'n2s'}
    DATE_KEY={'north':'s2nDate','south':'n2sDate'}

    # 时间
    date_ = content[DATE_KEY[model]]
    # vol为总和，model==north时，sh_hk为沪股通，反之为港股通（沪）
    temp_df = pd.DataFrame([item.split(",") for item in content[DATA_KEY[model]]],columns=['datetime', 'sh_hk','sz_hk','vol']) #

    # 原数据 日期和时间字段是分开的，且不带年份
    temp_df['datetime'] = pd.to_datetime(str(dt.now().year)+ '-' + date_+' '+temp_df['datetime'])
    temp_df= temp_df.set_index('datetime')
    try:
        temp_df= temp_df.astype('float64')
    except Exception as e:
        traceback.print_exception(type(e), e, sys.exc_info()[2])
        temp_df.to_excel('./error.xlsx', sheet_name='Sheet1', index=True)
        print(temp_df)
        

    temp_df['code']=CODE_KEY[model]   #兼容QA查询
    temp_df['model']=model
    temp_df['type']= QA.FREQUENCE.ONE_MIN
    temp_df["time_stamp"] = pd.to_datetime(temp_df.index).view(np.int64)//10**9     #兼容QA查询
    #print(temp_df)
    return temp_df


def save_north_line(north_df):
    """保存东方财富--北向数据 (基础函数)
    """    
    assert north_df is not None , 'north_df must be'
    assert len(north_df) >0 , 'north_df must not be 0 row'
    
    data = north_df.reset_index()
    freq = data.iloc[0].type
    
    
    if (freq==QA.FREQUENCE.DAY):
        coll = DATABASE.index_north_em_day
        coll.create_index([('code', pymongo.ASCENDING),("date_stamp", pymongo.ASCENDING)], unique=True)
    elif (freq==QA.FREQUENCE.ONE_MIN):
        coll = DATABASE.tmp_1min_index_north_em
        coll.create_index([('code', pymongo.ASCENDING),("time_stamp", pymongo.ASCENDING)], unique=True)

    else:
        raise Error('save_north_line: freq type error')

    # 查询是否新数据
    if (freq==QA.FREQUENCE.DAY):
        query_id = {
                        'code': data.iloc[0].code,
                        'date_stamp': {
                            '$in': data['date_stamp'].tolist()
                        }
                    }
    else:
        query_id = {
                        'code': data.iloc[0].code,
                        'time_stamp': {
                            '$in': data['time_stamp'].tolist()
                        }
                    }
    refcount = coll.count_documents(query_id)
    
    try:
        if refcount > 0:
            if (len(data) > 1):
                 # 删掉重复数据
                coll.delete_many(query_id)
                data = QA_util_to_json_from_pandas(data)
                coll.insert_many(data)
            else:
                 # 持续接收行情，更新记录
                if ('created_at' in data.columns):
                     data.drop('created_at', axis=1, inplace=True)
                data = QA_util_to_json_from_pandas(data)
                coll.replace_one(query_id, data[0])
        else:
             # 新 tick，插入记录
            data = QA_util_to_json_from_pandas(data)
            coll.insert_many(data)

    except Exception as e:
        if (data is not None):
            traceback.print_exception(type(e), e, sys.exc_info()[2])
            print(u'save_north_line failed!\n', e) 

            
def update_north_line(mode='fast'):
    """更新 东方财富--北向数据 (功能函数)
    """    
    print('start INDEX_NORTH_EM ====')
    save_north_line(fetch_north_day_from_eastmoney(mode=mode,model='north'))
    save_north_line(fetch_north_day_from_eastmoney(mode=mode,model='south'))
    print('finish INDEX_NORTH_EM ====')
    
    
def update_north_1min():
    """更新 东方财富--北向数据 (功能函数)
    """   
    print('start INDEX_NORTH_EM_1MIN ====')
    save_north_line(fetch_north_realtime_from_eastmoney(model='north'))
    save_north_line(fetch_north_realtime_from_eastmoney(model='south'))
    print('finish INDEX_NORTH_EM_1MIN ====')


def fetch_north_top10deal_from_eastmoney(type_='hk2sh',date_str='2021-09-28'):
    '''抓取东方财富--每日成交前10
        :param type_:{hk2sh|hk2sz|sh2hk|sz2hk} --指定交易方向，1沪股通  3深股通  2港股通(沪) 4港股通(深)
        注：北向和南向的货币单位不同，对比时注意汇率转换
        http://datacenter-web.eastmoney.com/api/data/v1/get?callback=jQuery112304439363764547424_1633887730442&sortColumns=RANK&sortTypes=1&pageSize=10&pageNumber=1&reportName=RPT_MUTUAL_TOP10DEAL&columns=ALL&source=WEB&client=WEB&filter=(MUTUAL_TYPE="001")(TRADE_DATE='2021-09-28')
    '''
    assert type_ in TOP10DEAL_TYPE, 'type_ error,must in {hk2sh|hk2sz|sh2hk|sz2hk}'

    def get_url():
        return "http://datacenter-web.eastmoney.com/api/data/v1/get"

    params = {
        'callback':'jQuery112304439363764547424_{:d}'.format(int(dt.utcnow().timestamp())),
        'sortColumns':'RANK',
        'sortTypes':1,
        'pageSize':10,
        'pageNumber':1,
        'reportName':'RPT_MUTUAL_TOP10DEAL',
        'columns':'ALL',
        'source':'WEB',
        'client':'WEB',
        'filter':'(MUTUAL_TYPE="{:s}")'.format(TOP10DEAL_TYPE2CODE[type_]) + "(TRADE_DATE=\'{:s}\')".format(date_str) 
    }

    json_data = request_json_get(get_url(), params, mode='jQuery', verbose=False)
#     print(json_data)
    try:
        if not json_data['success']:
            print('request no success, code:{:d},message:{:s}'.format(json_data['code'],json_data['message']))
            return None

        content_list = json_data['result']['data']
    except Exception as e:
        print(type(json_data))
        print(json_data)
        traceback.print_exception(type(e), e, sys.exc_info()[2])
        print(u'fetch_north_top10deal_from_eastmoney: failed\n', e) 
        return None

    temp_df = pd.DataFrame([item for item in content_list])
    temp_df = temp_df.rename(columns=TOP10DEAL_HEADDIC)
    temp_df['sse'] = temp_df['sse'].str.lower().str[-2:]
    temp_df['date'] = pd.to_datetime(temp_df['date']).dt.strftime('%Y-%m-%d')
    temp_df= temp_df.set_index('date')
    temp_df['type'] = temp_df['type'].map(TOP10DEAL_CODE2TYPE)
    
    percent_exchange = ['pct','main_ratio']
    temp_df[percent_exchange] = temp_df[percent_exchange] / 100
    
    model = 'north' if type_ in ['hk2sh','hk2sz'] else 'south'
    temp_df['model']=model

    temp_df["date_stamp"] = pd.to_datetime(temp_df.index).view(np.int64)//10**9     #兼容查询
    #print(temp_df)
    return temp_df

def save_top10deal(top10deal_df):
    """保存东方财富--北向十大成交记录 (基础函数)
    """    
    assert top10deal_df is not None , 'top10deal_df must be'
    assert len(top10deal_df) >0 , 'top10deal_df must not be 0 row'
    assert len(top10deal_df['type'].unique())==1, 'top10deal_df[type] must be unique in once task'
    
    data = top10deal_df.reset_index()
    freq = data.iloc[0].type
    
    coll = DATABASE.index_north_em_10top
    coll.create_index([("date_stamp", pymongo.ASCENDING)], unique=False)
    coll.create_index([('type', pymongo.ASCENDING),("date_stamp", pymongo.ASCENDING)], unique=False)
    coll.create_index([('model', pymongo.ASCENDING),("date_stamp", pymongo.ASCENDING)], unique=False)

    

    # 查询是否新数据
    query_id = {
                    'type': data.iloc[0].type,
                    'date_stamp': {
                        '$in': data['date_stamp'].tolist()
                    }
                }
            
    refcount = coll.count_documents(query_id)
    
    try:
        if refcount > 0:
            if (len(data) > 1):
                 # 删掉重复数据
                coll.delete_many(query_id)
                data = QA_util_to_json_from_pandas(data)
                coll.insert_many(data)
            else:
                 # 持续接收行情，更新记录
                if ('created_at' in data.columns):
                     data.drop('created_at', axis=1, inplace=True)
                data = QA_util_to_json_from_pandas(data)
                coll.replace_one(query_id, data[0])
        else:
             # 新 tick，插入记录
            data = QA_util_to_json_from_pandas(data)
            coll.insert_many(data)

    except Exception as e:
        if (data is not None):
            traceback.print_exception(type(e), e, sys.exc_info()[2])
            print(u'save_top10deal failed!\n', e) 
    
    
def update_deal_top10(verbose=False):
    print('start INDEX_NORTH_EM_10TOP ====')
    coll_top10_day = DATABASE.index_north_em_10top
    query = {'type':'hk2sh'}
    count = coll_top10_day.count_documents(query)
    end_date = str(now_time(separate_hour=22))[0:10]


    # 继续增量更新,同时防止初始化时出错
    if count > 0:
        # 接着上次获取的日期继续更新
        start_date = coll_top10_day.find_one(query,sort=[("date", -1)])['date']
    else:
        start_date = '2014-11-17'
        
    print('start_date',start_date)

    dates = date_range(start_date,end_date)
    print('dates',dates)
    sleep_params = np.random.exponential(scale=0.9,size=len(dates))+0.001
    print('sleep_params',sleep_params)
    for idx, d in enumerate(dates):
        time.sleep(sleep_params[idx])
        for type_ in TOP10DEAL_TYPE:
            df = fetch_north_top10deal_from_eastmoney(type_=type_,date_str=d)
            if not df is None:
                if len(df) != 0 :
                    if verbose:print('save', type_, d)
                    save_top10deal(df)
            else:
                print('jump',type_, d)
    print('finish INDEX_NORTH_EM_10TOP ====')
