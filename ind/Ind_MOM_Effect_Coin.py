
import sys
import os 
module_path = os.path.abspath(os.path.join('..')) 
if module_path not in sys.path: 
    sys.path.append(module_path)


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tools.Sample_Tools as smpl

from base.JuUnits import excute_for_multidates


# %load_ext autoreload
# %autoreload 2
# %aimport Analysis_Funs,Ind_Model_Base


def get_interday_fluctuation_reverse(stock_df, cur_ret, turnover):
    #日间反转-波动翻转
    # 1）每天计算所有股票的日间收益率。
    # 2）每月月底计算最近20天的日间收益率的均值和标准差，作为当月的日间收益率和日间波动率。
    # 3）比较每只股票的日间波动率与市场截面均值的大小关系，将日间波动率小于市场均值的股票，
    # 视为“硬币”型股票，由于未来其发生动量效应的概率更大，因此我们将其当月日间收益率乘以-1；
    # 而日间波动率大于市场均值的股票，视为“球队”型股票，其未来将发生反转效应的概率更大，
    # 其因子值保持不变。
    # 我们将变换后的因子作为修正后的新反转因子，记为“日间反转-波动翻转”因子
    
    # cur_ret = smpl.get_current_return(data_,'close')
    # turnover = data_['volume'] / (data_['lshares'] *100) # 手/万股

    ret_std20 = excute_for_multidates(cur_ret, lambda x:x.rolling(20).std(), level=1)
    market_std = excute_for_multidates(ret_std20, lambda x:x.mean(), level=0)
    condition = excute_for_multidates(ret_std20, lambda x:x<market_std[x.index.get_level_values(0)[0]], level=0)
    interday_fluctuation_reverse = cur_ret*condition.map({True:-1,False:1})
    interday_fluctuation_reverse = excute_for_multidates(interday_fluctuation_reverse, lambda x:x.rolling(20).mean(), level=1)
    interday_fluctuation_reverse.name='interday_fluctuation_reverse'
    return interday_fluctuation_reverse

def get_interday_turnover_reverse(stock_df, cur_ret, turnover):
    # “日间反转-换手翻转”
    # 因子的定义接下来我们使用换手率的变化量这一指标，在日频维度上，寻找可能发生动量效应的股票，并将其这一天的日间涨跌幅翻转过来。
    # 1）计算每支股票t日换手率与t-1日换手率的差值，作为t日换手率的变化量。
    # 2）将每只股票的换手率变化量与当日全市场的换手率变化量的均值做比较，我们认为换手率变化量高于市场均值的股票为“球队”型股票，其未来将大概率发生反转效应；换手率变化量低于市场均值的，为“硬币”股票，未来将大概率发生动量效应。
    # 3）我们计算每只股票t日的日间收益率，将“硬币”型股票的日间收益率乘以-1，而“球队”型股票的日间收益率保持不变。记变化后的日间收益率为“翻转收益率”。
    # 4）每月月底，计算最近20天的“翻转收益率”的均值，我们将变换后的因子作为经修正后的新反转因子，记为本月的“日间反转-换手翻转”因子。
    turnover_dif = excute_for_multidates(turnover, lambda x:x.diff(), level=1)
    market_turnover_dif = excute_for_multidates(turnover_dif, lambda x:x.mean(), level=0)
    condition = excute_for_multidates(turnover_dif, lambda x:x<market_turnover_dif[x.index.get_level_values(0)[0]], level=0)
    interday_turnover_reverse = cur_ret*condition.map({True:-1,False:1})
    interday_turnover_reverse = excute_for_multidates(interday_turnover_reverse, lambda x:x.rolling(20).mean(), level=1)
    interday_turnover_reverse.name='interday_turnover_reverse'
    return interday_turnover_reverse

def get_intraday_fluctuation_reverse(stock_df, turnover):
    # 日内反转-波动翻转
    # 1）每天计算每只股票的日内收益率。
    # 2）每月月底计算最近20天的日内收益率的均值和标准差，作为当月的日内收益率和日内收益率的波动率。
    # 3）比较每只股票的日内收益率的波动率与市场截面均值的大小关系，将日内收益率的波动率小于市场均值的股票，视为“硬币”型股票，其未来发生动量效应的概率更大，
    # 因此我们将其当月日内收益率乘以-1；而日内收益率的波动率大于市场均值的股票，视为“球队”型股票，其未来将发生反转效应的概率更大，其当月日内收益率保持不变。
    # 我们将变换后的因子作为修正后的新日内反转因子，记为“日内反转-波动翻转”因子
    intra_ret = stock_df['close'] / stock_df['open'] -1
    intra_ret_std20 = excute_for_multidates(intra_ret, lambda x:x.rolling(20).std(), level=1)

    intra_market_std = excute_for_multidates(intra_ret_std20, lambda x:x.mean(), level=0)
    condition = excute_for_multidates(intra_ret_std20, lambda x:x<intra_market_std[x.index.get_level_values(0)[0]], level=0)
    intraday_fluctuation_reverse = intra_ret*condition.map({True:-1,False:1})
    intraday_fluctuation_reverse = excute_for_multidates(intraday_fluctuation_reverse, lambda x:x.rolling(20).mean(), level=1)
    intraday_fluctuation_reverse.name='intraday_fluctuation_reverse'
    return intraday_fluctuation_reverse

def get_intraday_turnover_reverse(stock_df, turnover):
    # 日内反转-换手翻转
    intra_ret = stock_df['close'] / stock_df['open'] -1
    turnover_dif = excute_for_multidates(turnover, lambda x:x.diff(), level=1)
    market_turnover_dif = excute_for_multidates(turnover_dif, lambda x:x.mean(), level=0)
    condition = excute_for_multidates(turnover_dif, lambda x:x<market_turnover_dif[x.index.get_level_values(0)[0]], level=0)
    intraday_turnover_reverse = intra_ret*condition.map({True:-1,False:1})
    intraday_turnover_reverse = excute_for_multidates(intraday_turnover_reverse, lambda x:x.rolling(20).mean(), level=1)
    intraday_turnover_reverse.name='intraday_turnover_reverse'
    return intraday_turnover_reverse

def get_overnight_fluctuation_reverse(stock_df):
# 隔夜反转-波动翻转
    overnight_ret = excute_for_multidates(stock_df, lambda x:(x['open']/x['close'].shift(1))-1, level='code')
    overnight_ret_std20 = excute_for_multidates(overnight_ret, lambda x:x.rolling(20).std(), level=1)

    overnight_market_std = excute_for_multidates(overnight_ret_std20, lambda x:x.mean(), level=0)
    condition = excute_for_multidates(overnight_ret_std20, lambda x:x<overnight_market_std[x.index.get_level_values(0)[0]], level=0)
    overnight_fluctuation_reverse = overnight_ret_std20*condition.map({True:-1,False:1})
    overnight_fluctuation_reverse = excute_for_multidates(overnight_fluctuation_reverse, lambda x:x.rolling(20).mean(), level=1)
    overnight_fluctuation_reverse.name='overnight_fluctuation_reverse'
    return overnight_fluctuation_reverse

def get_overnight_turnover_reverse(stock_df, turnover):
    # 隔夜反转-换手翻转
    overnight_ret = excute_for_multidates(stock_df, lambda x:(x['open']/x['close'].shift(1))-1, level='code')
    turnover_dif = excute_for_multidates(turnover, lambda x:x.diff(), level=1)
    market_turnover_dif = excute_for_multidates(turnover_dif, lambda x:x.mean(), level=0)
    condition = excute_for_multidates(turnover_dif, lambda x:x<market_turnover_dif[x.index.get_level_values(0)[0]], level=0)
    overnight_turnover_reverse = overnight_ret*condition.map({True:-1,False:1})
    overnight_turnover_reverse = excute_for_multidates(overnight_turnover_reverse, lambda x:x.rolling(20).mean(), level=1)
    overnight_turnover_reverse.name='overnight_turnover_reverse'
    return overnight_turnover_reverse
