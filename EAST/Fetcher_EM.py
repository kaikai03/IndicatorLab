import os
import sys
import time

import pandas as pd

module_path = os.path.abspath(os.path.join('..')) 
if module_path not in sys.path: 
    sys.path.append(module_path)
    

from QUANTAXIS.QAUtil import (
    DATABASE)

import QUANTAXIS as QA

import tools.Sample_Tools as smpl
from base.JuUnits import (
    fetch_index_day_common,
    now_time_tradedate,
)
# from QUANTAXIS.QAUtil.QADate_trade import (
#     QA_util_if_trade)

from Crawler_Block_East import (
    EM_MONEYFOLOW_CNKEY_DIC,
    fetch_block_moneyflow_realtime_from_eastmoney,
    fetch_block_kline_realtime_from_eastmoney
)

from Crawler_North_East import (
    update_north_line,
)


def fetch_all_blocks(hy_source='concept'):
    return smpl.get_all_blocks(hy_source=hy_source, collections=DATABASE.stock_block_em)

def fetch_blocks_view(hy_source='industry'):
    return smpl.get_blocks_view(hy_source=hy_source, collections=DATABASE.stock_block_em)

def fetcher_block_info(code, hy_source='industry', collections=DATABASE.stock_block_em):
    code_ = code
    if isinstance(code, str):
        code_ = [code]
    try:
        res = QA.QA_fetch_stock_block(code=code_, collections=collections)
    except Exception as e:
        print(e,'get_stock_blockname ：code error')
        return None
    return res[res['type']==hy_source]

def fetch_block_index_day(block_code='BK0428',start='2021-09-02',end='2021-10-13'):
    return QA.QA_fetch_stock_day_adv(code=block_code,start=start,end=end,collections=DATABASE.stock_block_em_day)

def fetch_block_index_min(block_code='BK0428',start='2021-09-02',end='2021-09-03',frequence=QA.FREQUENCE.HOUR):
    return QA.QA_fetch_index_min_adv(code=block_code,start=start,end=end,frequence=frequence,collections=DATABASE.stock_block_em_min)

def fetch_block_index_1min(block_code='BK0428',start='2021-09-02',end='2021-10-12'):
    return QA.QA_fetch_index_min_adv(code=block_code,start=start,end=end,frequence=QA.FREQUENCE.ONE_MIN,collections=DATABASE.tmp_1min_stock_block_em)

def fetch_block_moneyflow_realtime(block_code='BK0428'):
    se = fetch_block_moneyflow_realtime_from_eastmoney(block_code)
    if not se is None:
        return se.rename(EM_MONEYFOLOW_CNKEY_DIC)
    return se

def fetch_stock_codelist_by_blockname(block_name='民航机场'):
    series = pd.Series(smpl.get_codes_from_blockname(block_name, collections=DATABASE.stock_block_em))
    series.index = series.to_list()
    ns = QA.QA_fetch_stock_name(series.to_list())
    dic = ns['name'].to_dict()
    return series.rename(dic)

def fetch_north_deal_day(direct='north_', start='2021-10-01', end='2021-11-18'):
    return QA.QA_fetch_index_day_adv(direct,start,end,collections=DATABASE.index_north_em_day)

def fetch_north_deal_1min(direct='north_', start='2021-10-01', end='2021-11-18', ):
    df = QA.QA_fetch_index_min(direct,start,end, frequence=QA.FREQUENCE.ONE_MIN, format='pd',collections=DATABASE.tmp_1min_index_north_em)
    if df is None:
        return df
    return df.reset_index(drop=True).set_index(['datetime','code'])

def fetch_north_10top_direct(direct='north', start='2021-10-01', end='2021-11-18', ):
    return fetch_index_day_common('model',direct,start,end,collections=DATABASE.index_north_em_10top)

def fetch_north_10top_type(type_=['hk2sz','hk2sh'], start='2021-10-01', end='2021-11-18', ):
    return fetch_index_day_common('type',type_,start='2014-11-17',end='2021-11-18',collections=DATABASE.index_north_em_10top)
