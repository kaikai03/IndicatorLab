
import sys

import os 
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
import datetime
import pymongo
import traceback
from tqdm.autonotebook import trange, tqdm

from base.JuNetwork import request_json_get


EM_KEY_DIC = {
    'f2':'close', #最新价
    'f3':'pct', #涨跌幅
    'f4':'change', #涨跌额
    'f5':'vol', #成交量(手)
    'f6':'amount', #成交额
    'f7':'amplitude', #振幅
    'f8':'turnoverrate', #换手率
    'f9':'ttm', #市盈率(动态)
    'f10':'volrate', #量比
    'f11':'pct5m', #五分钟涨跌
    'f12':'code', #概念代码
    
    'f13':'SSE', #交易所 0sz,1sh

    'f14':'name', #板块/stock名称
    'f15':'high', #最高
    'f16':'low', #最低
    'f17':'open', #今开
    'f18':'preclose', #昨收
    'f20':'marketvalue', #总市值
    'f21':'flowcapitalvalue', #流通市值
    'f22':'speed', #涨速
    'f23':'pb', #市净率
    'f24':'pct60d', #60日涨跌幅
    'f25':'pct360d', #年初至今涨跌幅
    
    'f45':'JLV', #净利润
    
    'f62':'BalFlowMainSub',  #主力净流入
    'f184':'ratioMainSub', #主力净占比
    
    'f104':'upcount', #上涨家数
    'f105':'downcount', #下跌家数
    'f115':'pe', #市盈率
    'f128':'uplead', #领涨股票
    'f136':'upleadpct', #领涨涨跌幅
    'f140':'upcode', #领涨code
    
    'f207':'downlead', #领跌股票
    'f208':'downcode', #领跌code
    'f222':'downleadpct', #领跌涨跌幅
    

    'f124':'timestamp', #最新行情时间

    }

EM_MONEYFOLOW_KEY_DIC={
    'f57':'code', # 版块代码
    'f135':'zllr', # 主力流入
    'f136':'zllc', # 主力流出
    'f137':'zljlr', # 主力净流入
    'f138':'cddlr', # 超大单流入
    'f139':'cddlc', # 超大单流入
    'f140':'cddjlr', # 超大单净流入
    'f141':'ddlr', # 大单流入
    'f142':'ddlc', # 大单流出
    'f143':'ddjlr', # 大单净流入
    'f144':'zdlr', # 中单流入
    'f145':'zdlc', # 中单流出
    'f146':'zdjlr', # 中单净流入
    'f147':'xdlr', # 小单流入
    'f148':'xdlc', # 小单流出
    'f149':'xdjlr', # 小单净流入
    'f193':'zlzb', # 主力净占比
    'f194':'cddjzb', # 超大单净占比
    'f195':'ddjzb', # 大单净占比
    'f196':'zdjzb', # 中单净占比
    'f197':'zdjzb', # 小单净占比
    'f152':'decimal', # 小数位数标识
}

EM_MONEYFOLOW_CNKEY_DIC={
    'code':'版块代码',
    'zllr':'主力流入',
    'zllc':'主力流出',
    'zljlr':'主力净流入',
    'cddlr':'超大单流入',
    'cddlc':'超大单流入',
    'cddjlr':'超大单净流入',
    'ddlr':'大单流入',
    'ddlc':'大单流出',
    'ddjlr':'大单净流入',
    'zdlr':'中单流入',
    'zdlc':'中单流出',
    'zdjlr':'中单净流入',
    'xdlr':'小单流入',
    'xdlc':'小单流出',
    'xdjlr':'小单净流入',
    'zlzb':'主力净占比',
    'cddjzb':'超大单净占比',
    'ddjzb':'大单净占比',
    'zdjzb':'中单净占比',
    'zdjzb':'小单净占比',
    'decimal':'小数位数标识'
}

EM_STAT_DIC = {
     "-2":  "已收盘",
     "-1":  "停牌",
     "0":  "交易中",
     "1":  "已收盘",
     "2":  "午间休市",
     "3":  "已休市",
     "4":  "未开盘",
     "5":  "已收盘",
     "6":  "已收盘",
     "7":  "已收盘",
     "8":  "暂停交易",
     "9":  "暂停交易",
     "10":  "暂停交易",
     "11":  "暂停交易",
     "12": "未上市"
    }



EM_KLINE_TAG_LIST = [
    'datetime',
    'open',
    'close',
    'high',
    'low',
    'vol',       # 成交量
    'amount',    # 成交额
    'amplitude', # 振幅
    'spreadrate', # 涨跌幅
    'spread',    # 涨跌额
    'turnover'    # 换手
]


EM_KLINE_REALTIME_TAG_LIST = [
    'datetime',
    'open',
    'close',
    'high',
    'low',
    'vol',
    'amount',
    'avg'
]

EM_FREQUENCE_DIC ={
    QA.FREQUENCE.MONTH:'103',
    QA.FREQUENCE.WEEK:'102',
    QA.FREQUENCE.DAY:'101',
    QA.FREQUENCE.HOUR:'60',
    QA.FREQUENCE.THIRTY_MIN:'30',
    QA.FREQUENCE.FIFTEEN_MIN:'15',
    QA.FREQUENCE.FIVE_MIN:'5'
}


def fetch_stock_block_list_from_eastmoney(pagenumber=1, pagelimit=400, model='concept'):
    '''抓取东方财富--板块列表（基础函数）
    '''
    def get_random_stock_block_url():
            url = "http://{:d}.push2.eastmoney.com/api/qt/clist/get".format(randint(1, 99))
            return url

    params = {
        "pn": pagenumber,
        "pz": pagelimit,
        "np": '1',
        "fltt": '2',
        "invt": '2',
        'fid': 'f3',
        'fs': 'm:90+t:3+f:!50' if (model=='concept') else 'm:90+t:2+f:!50',
        "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f33,f11,f62,f128,f136,f115,f152,f124,f107,f104,f105,f140,f141,f207,f208,f209,f222",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "cb": "jQuery1124023986497915529914_{:d}".format(int(dt.utcnow().timestamp())),
        "_": int(time.time() * 1000),
    }

    json_data = request_json_get(get_random_stock_block_url(), params, mode='jQuery', verbose=False)
    
    try:
        if (json_data["data"] is None):
            print('json_data["data"] is None')
            return None
        content_list = json_data["data"]["diff"]
    except Exception as e:
        print(type(json_data))
        print(json_data)
        traceback.print_exception(type(e), e, sys.exc_info()[2])
        print(u'fetch_stock_block_list_from_eastmoney: failed\n', e) 
        return None

#         print(json_data)

    temp_df = pd.DataFrame([item for item in content_list])
#     print(temp_df)
    ret_stock_block_list = temp_df.rename(columns=EM_KEY_DIC)
    percent_exchange = ['pct','amplitude','turnoverrate','pct5m','speed','pct60d','pct360d','upleadpct','downleadpct']
    ret_stock_block_list[percent_exchange] = ret_stock_block_list[percent_exchange] / 100
    ret_stock_block_list.replace('-', 0, inplace=True) 
    ret_stock_block_list['resource'] = 'eastmoney'
    ret_stock_block_list['date'] = ret_stock_block_list['timestamp'].apply(lambda x : pd.to_datetime(x,utc=True, unit='s').tz_convert('Asia/Shanghai').strftime("%Y-%m-%d %H:%M:%S"))
    ret_stock_block_list['type'] = 'concept' if (model=='concept') else 'industry'
    return ret_stock_block_list


def fetch_stock_block_components_from_eastmoney(concept='BK0731',
                                  pagenumber=1,
                                  pagelimit=400):
    '''抓取东方财富板块概念--成分数据（基础函数）
    http://28.push2.eastmoney.com/api/qt/clist/get?cb=jQuery112405892331704393758_1629080255227&pn=2&pz=20&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=b:BK0981+f:!50&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152,f45&_=1629080255234    
    '''
    def get_random_stock_block_components_url():
        url = "http://{:d}.push2.eastmoney.com/api/qt/clist/get".format(randint(1, 99))
        return url
    
    params = {
        "pn": pagenumber,
        "pz": pagelimit,
        "po": '1',
        "np": '1',
        "fltt": '2',
        "invt": '2',
        'fid': 'f3',
        'fs': 'b:{:s}+f:!50'.format(concept),
        "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152,f45",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "cb": "jQuery1124023986497915529914_{:d}".format(int(dt.utcnow().timestamp())),
        "_": int(time.time() * 1000),
    }
    
    json_data = request_json_get(get_random_stock_block_components_url(), params, mode='jQuery', verbose=False)
    
    try:
        if (json_data["data"] is None):
            print('json_data["data"] is None')
            return None
        content_list = json_data["data"]["diff"]
    except Exception as e:
        print(type(json_data))
        print(json_data)
        traceback.print_exception(type(e), e, sys.exc_info()[2])
        print(u'fetch_stock_block_components_from_eastmoney: failed\n', e) 
        return None
        #print(json_data)

    temp_df = pd.DataFrame([item for item in content_list])
    # #print(temp_df)

    ret_stock_concept_components = temp_df.rename(columns=EM_KEY_DIC)

    ret_stock_concept_components.replace('-', 0, inplace=True) 
    percent_exchange = ['pct','amplitude','turnoverrate','pct5m','speed','pct60d','pct360d']
    ret_stock_concept_components[percent_exchange] = ret_stock_concept_components[percent_exchange] / 100
    

    return ret_stock_concept_components


def fetch_block_kline_from_eastmoney(block_code: str="BK0428",  freq=QA.FREQUENCE.DAY) -> pd.DataFrame:
    """获取板块k线（基础函数）
    http://push2his.eastmoney.com/api/qt/stock/kline/get?cb=jQuery112404939798105940868_1629173273789&secid=90.BK0917&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=60&fqt=0&beg=19900101&end=20220101&_=1629173274401
    """
    assert freq in [QA.FREQUENCE.DAY, QA.FREQUENCE.HOUR, QA.FREQUENCE.FIVE_MIN], 'freq only support DAY|HOUR|FIVE_MIN（自我规制）'
    url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "cb": "jQuery1124005797095004732822_{:d}".format(int(dt.utcnow().timestamp())),
        "secid": f"90.{block_code}",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1,f2,f3,f4,f5",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "fqt": "0",
        "klt": "101" if freq == QA.FREQUENCE.DAY else ("15" if freq == QA.FREQUENCE.FIFTEEN_MIN else "60"),
        'beg':'19900101',
        'end': '20250101',
        "_": int(time.time() * 1000),
    }
    
    json_data = request_json_get(url, params, mode='jQuery', verbose=False)

    try:
        if (json_data["data"] is None):
            print('json_data["data"] is None')
            return None
        content_list = json_data["data"]["klines"]
    except Exception as e:
        print(type(json_data))
        print(json_data)
        traceback.print_exception(type(e), e, sys.exc_info()[2])
        print(u'fetch_block_kline_from_eastmoney: failed\n', e) 
        return None
    
    #     print(json_data)

    temp_df = pd.DataFrame([item.split(",") for item in content_list])

    temp_df.columns = EM_KLINE_TAG_LIST

    temp_df.loc[:, EM_KLINE_TAG_LIST[1:]] = temp_df[EM_KLINE_TAG_LIST[1:]].astype(np.float64)
    temp_df['code'] = block_code

    percent_exchange = ['amplitude','spreadrate','turnover']
    temp_df[percent_exchange] = temp_df[percent_exchange] / 100

    temp_df['type'] = freq
    temp_df['date'] = pd.to_datetime(temp_df['datetime']).dt.strftime('%Y-%m-%d')
    temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])
    temp_df['datetime'] = temp_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')  #兼容day和min
    temp_df['datetime'] = pd.to_datetime(temp_df['datetime']) #为了方便重采样，入库前会被QA的tojson转回字符串
    # GMT+0 String 转换为 UTC Timestamp
    temp_df['time_stamp'] = pd.to_datetime(temp_df['datetime']).view(np.int64)//10**9   #兼容QA查询
    temp_df["date_stamp"] = pd.to_datetime(temp_df['date']).view(np.int64)//10**9     #兼容QA查询

    #print(temp_df)
    return temp_df.set_index('datetime')  #为了方便重采样，入库时会reset

    
    
def fetch_block_kline_realtime_from_eastmoney(block_code='BK0428'):
    '''抓取东方财富板块概念--日内分时线（基础函数）
        http://push2.eastmoney.com/api/qt/stock/trends2/get?cb=jQuery112409178740256648212_1633597315058&secid=90.BK0428&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13&fields2=f51,f52,f53,f54,f55,f56,f57,f58&iscr=0&ndays=1&_=1633597317138
    '''
        
    def get_url():
        return "http://push2.eastmoney.com/api/qt/stock/trends2/get"
    
    params = {
        'cb': 'jQuery1124021715896797765277_{:d}'.format(int(dt.utcnow().timestamp())),
        'secid': '90.{:s}'.format(block_code),
        'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
        'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58',
        'iscr': 0,
        'ndays': 1,
        '_': int(time.time() * 1000)
    }
    
    json_data = request_json_get(get_url(), params, mode='jQuery', verbose=False)

    try:
        if (json_data["data"] is None):
            print('json_data["data"] is None')
            return None
        content_list = json_data["data"]["trends"]
    except Exception as e:
        print(type(json_data))
        print(json_data)
        traceback.print_exception(type(e), e, sys.exc_info()[2])
        print(u'fetch_block_kline_realtime_from_eastmoney: failed\n', e) 
        return None
    
    #print(json_data)

    temp_df = pd.DataFrame([item.split(",") for item in content_list])
    temp_df.columns = EM_KLINE_REALTIME_TAG_LIST
    temp_df.loc[:, EM_KLINE_REALTIME_TAG_LIST[1:]] = temp_df[EM_KLINE_REALTIME_TAG_LIST[1:]].astype(np.float64)
    temp_df['code'] = block_code
    temp_df['type'] = '1min'
    temp_df['date'] = pd.to_datetime(temp_df['datetime']).dt.strftime('%Y-%m-%d')
    temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])
    temp_df['datetime'] = temp_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')  #兼容day和min
    temp_df['datetime'] = pd.to_datetime(temp_df['datetime']) #为了方便重采样，入库前会被QA的tojson转回字符串
    # GMT+0 String 转换为 UTC Timestamp
    temp_df['time_stamp'] = pd.to_datetime(temp_df['datetime']).view(np.int64)//10**9   #兼容QA查询
    temp_df["date_stamp"] = pd.to_datetime(temp_df['date']).view(np.int64)//10**9     #兼容QA查询

    #print(temp_df)
    return temp_df.set_index('datetime') #为了方便重采样，入库时会reset

    

def fetch_block_moneyflow_realtime_from_eastmoney(block_code='BK0428'):
    '''抓取东方财富板块概念实时资金流（基础函数）
        http://push2.eastmoney.com/api/qt/stock/get?secid=90.BK0428&ut=bd1d9ddb04089700cf9c27f6f7426281&fields=f57,f135,f136,f137,f138,f139,f140,f141,f142,f143,f144,f145,f146,f147,f148,f149,f193,f194,f195,f196,f197,f152&cb=jQuery1124021715896797765277_1633592123562&_=1633592123570
    '''
        
    def get_url():
        return "http://push2.eastmoney.com/api/qt/stock/get"
    
    params = {
        'secid': '90.{:s}'.format(block_code),
        'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
        'fields': 'f57,f135,f136,f137,f138,f139,f140,f141,f142,f143,f144,f145,f146,f147,f148,f149,f193,f194,f195,f196,f197,f152',
        'cb': 'jQuery1124021715896797765277_{:d}'.format(int(dt.utcnow().timestamp())),
        '_': int(time.time() * 1000)
    }
    
    json_data = request_json_get(get_url(), params, mode='jQuery', verbose=False)

    try:
        if (json_data["data"] is None):
            print('json_data["data"] is None')
            return None
        content_dic = json_data["data"]
    except Exception as e:
        print(type(json_data))
        print(json_data)
        traceback.print_exception(type(e), e, sys.exc_info()[2])
        print(u'fetch_block_moneyflow_realtime_from_eastmoney: failed\n', e) 
        return None
    
    #print(json_data)
    temp_sr = pd.Series(content_dic)
    # #print(temp_df)
    ret_moneyflow_realtime = temp_sr.rename(EM_MONEYFOLOW_KEY_DIC)
    return ret_moneyflow_realtime


def fetch_east_stock_block_from_eastmoney(verbose=False):
    '''获取概念板块数据以及对应的成分（功能组合函数）
        其他基础函数功能的执行
       :param verbose:{bool} --是否打印数据获取过程。(default: False)
    '''
    print('Now Fetching eastmoney concept block ====')
    concept = fetch_stock_block_list_from_eastmoney(model='concept')
    if concept is None:
        raise 'em接口出错，concept获取失败'
    
    print('Now Fetching eastmoney industry block ====')
    industry = fetch_stock_block_list_from_eastmoney(model='industry')
    if industry is None:
        raise 'em接口出错，industry获取失败'
    
    concept = concept[['code', 'name','type','resource']]
    industry = industry[['code', 'name','type','resource']]
    
    print('Now Fetching eastmoney stock components ====')
    infos_all = []
    total=len(concept)+len(industry)
    
    # 生成随机延迟
    sleep_params = np.random.exponential(scale=0.3,size=total)+0.001
    
    for idx, item in tqdm(pd.concat([concept,industry]).iterrows(),total=total):
        if verbose:
            print(item['name'],item['code'])
        stocks = fetch_stock_block_components_from_eastmoney(concept=item['code'])
        infos = stocks[['code','name']].copy()
        infos['blockname'] = item['name']
        infos['blockcode'] = item['code']
        infos['type'] = item['type']
        infos['resource'] = item['resource']
        infos_all.append(infos)
        time.sleep(sleep_params[idx])

    infos_df = pd.concat(infos_all,axis=0)
    return infos_df


def update_east_stock_block(blocks:pd.DataFrame=None,client=DATABASE):
    '''重新保存板块成分（入库函数）
       :param blocks:{pd.DataFrame} 
              --不为空时，将该对象入库，否则直接获取。(default: None)
    '''
    blocks_ = blocks
    if blocks_ is None:
        blocks_ = fetch_east_stock_block_from_eastmoney(verbose=False)
        
    if blocks_ is None or len(blocks_)==0:
        raise 'em接口出错，未获取到任何数据'
    
    client.drop_collection('stock_block_em')
    coll = client.stock_block_em
    coll.create_index('code')
    coll.create_index('name')
    
    try:
        print('Now Saving eastmoney EM_STOCK_BlOCK ====')
        coll.insert_many(
            QA_util_to_json_from_pandas(blocks_)
        )
    except Exception as e:
        print(e)
        print(" Error save_east_stock_block_infos exception!")
        
    print('finish EM_STOCK_BlOCK ====')
    

def save_stock_block_kline(block_kline_df):
    """保存东方财富股票概念板块K线数据（功能函数）
        注：1min数据日内产生，有可能会有遗漏，当临时数据，故另外存放，随时丢弃。
    """    
    assert block_kline_df is not None , 'block_kline_df must be'
    assert len(block_kline_df) >0 , 'block_kline_df must not be 0 row'
    
    data = block_kline_df.reset_index()
    freq = data.iloc[0].type
    
    assert freq in [QA.FREQUENCE.DAY, QA.FREQUENCE.HOUR, QA.FREQUENCE.FIVE_MIN, QA.FREQUENCE.ONE_MIN], 'freq only support DAY|HOUR|FIVE_MIN|ONE_MIN（自我规制）'
    
    if (freq==QA.FREQUENCE.DAY):
        coll = DATABASE.stock_block_em_day
        coll.create_index([('code', pymongo.ASCENDING),("date_stamp", pymongo.ASCENDING)], unique=True)
    elif (freq==QA.FREQUENCE.ONE_MIN):
        coll = DATABASE.tmp_1min_stock_block_em
        coll.create_index([('code', pymongo.ASCENDING),("date", pymongo.ASCENDING)], unique=False)
        coll.create_index([('code', pymongo.ASCENDING),("time_stamp", pymongo.ASCENDING)], unique=True)
    else:
        coll = DATABASE.stock_block_em_min
        coll.create_index([('code', pymongo.ASCENDING), ("type", pymongo.ASCENDING), ("date", pymongo.ASCENDING)], unique=False)
        coll.create_index([('code', pymongo.ASCENDING), ("type", pymongo.ASCENDING), ("time_stamp", pymongo.ASCENDING)], unique=True)

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
                        'type': freq,
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
            print(u'save_stock_block_kline failed!!\n', e) 

            
def update_all_block_kline(error_threhold=5,verbose=False):
#     assert freq in [QA.FREQUENCE.DAY, QA.FREQUENCE.HOUR, QA.FREQUENCE.FIVE_MIN], 'freq only support DAY|HOUR|FIVE_MIN（自我规制）'
    """更新数据库k线数据（功能函数）
        注：不含1min日内线，1min为临时数据
    """   
    print('Now Fetching eastmoney block kline====')
    all_code = fetch_eastmoney_block_code()
    total = len(all_code)
    assert total>0,'error: fetch_eastmoney_block_code()未取得数据 '
    
    error_count = 0
    
    # 生成随机延迟
    sleep_params = np.random.exponential(scale=0.3,size=total)+0.001
    
    for idx, block_code in tqdm(enumerate(all_code) ,total=total):
        if error_count > error_threhold:
            raise '累计错误超过额定次数，kline更新终止 ========='
        
        if verbose:
            print('start:',idx, block_code)
        
        time.sleep(sleep_params[idx])
        
        k_day_df = fetch_block_kline_from_eastmoney(block_code,freq=QA.FREQUENCE.DAY)
        if k_day_df is None:
            print(("%s day未取到数据") % block_code) 
            error_count+=1
            continue
        save_stock_block_kline(k_day_df)
        time.sleep(sleep_params[idx]/randint(1, 3))
        
        k_hour_df = fetch_block_kline_from_eastmoney(block_code,freq=QA.FREQUENCE.HOUR)
        if k_hour_df is None:
            print(("%s hour未取到数据") % block_code) 
            error_count+=1
            continue
        save_stock_block_kline(k_hour_df)
        time.sleep(sleep_params[idx]/randint(1, 4))
        
        k_5min_df = fetch_block_kline_from_eastmoney(block_code,freq=QA.FREQUENCE.FIVE_MIN)
        if k_5min_df is None:
            print(("%s 5min未取到数据") % block_code) 
            error_count+=1
            continue
        save_stock_block_kline(k_5min_df)
        
#         测试
#         if idx == 2:
#             return
        
    print('Now Finishing eastmoney block kline====')
    
    
def update_1min_block_kline(error_threhold=5,verbose=False):
    """获取/更新 今日 1min 板块k线 （功能函数）
       注：1min为临时数据，不完整且随时丢弃
    """   
    print('Now Fetching eastmoney block 1min_tmp kline====')
    all_code = fetch_eastmoney_block_code()
    total = len(all_code)
    assert total>0,'error: fetch_eastmoney_block_code()未取得数据 '
    
    error_count = 0
    
    # 生成随机延迟
    sleep_params = np.random.exponential(scale=0.3,size=total)+0.001
    
    for idx, block_code in tqdm(enumerate(all_code) ,total=total):
        if error_count > error_threhold:
            raise '累计错误超过额定次数，kline更新终止 ========='
        
        if verbose:
            print('start:',idx, block_code)
        
        time.sleep(sleep_params[idx])
        
        k_df = fetch_block_kline_realtime_from_eastmoney(block_code)
        if k_df is None:
            print(("%s 1min未取到数据") % block_code) 
            error_count+=1
            continue
        save_stock_block_kline(k_df)
        time.sleep(sleep_params[idx]/randint(1, 3))
        
        save_stock_block_kline(k_df)
        
#         测试
#         if idx == 2:
#             return
        
    print('Now Finishing eastmoney block 1min_tmp kline====')
    
    
def fetch_eastmoney_block_code(blockname:[list,str]='all',collections=DATABASE.stock_block_em):
    '''读取板块代码列表（功能函数）
       :param blockname:{list|str} 
              --板块名称，all或none时返回全部代码。(default: all)
    '''
    if blockname=='all' or blockname is None:
         return[
            item['blockcode']
            for item in collections.aggregate([
                {"$group": {"_id": "$blockcode", "count": {"$sum": 1}, }},
                {"$project": {"blockcode": "$_id",  "_id": 0, }}
            ])
        ]
    
    blockname_ = blockname
    if not isinstance(blockname_,list):
        blockname_ = [blockname]
    
    return[
        item['blockcode']
        for item in collections.aggregate([
            {"$match": {"blockname": {"$in": blockname_}}},
            {"$group": {"_id": "$blockcode", "count": {"$sum": 1}, }},
            {"$project": {"blockcode": "$_id",  "_id": 0, }}
        ])
    ]
