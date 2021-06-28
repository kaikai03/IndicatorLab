
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as st
import statsmodels.api as sm

from sklearn import linear_model

def get_rank_ic(factor_standardized, ret_forward):
    """
       计算因子的信息系数
        :param factor_standardized: {multiIndex[date,code]} --标准化后的因子值
        :param ret_forward: {multiIndex[date,code]} --下一期的股票收益率
        :return {pd.Series}
    """
    
    assert len(factor_standardized.index.get_level_values(1).unique()) > 4, '股票数量必须大于4，否则没啥意义啊'

    #index取交集
    common_index = factor_standardized.index.get_level_values(0).unique().intersection(ret_forward.index.get_level_values(0).unique())
    ic_data = pd.Series([None]*len(common_index),index=common_index)
    
    df = pd.DataFrame({'factor_standardized':factor_standardized, 'ret_f':ret_forward})
    df = df[~pd.isnull(df['factor_standardized'])][~pd.isnull(df['ret_f'])]

    # 计算相关系数
    for dt in df.index.get_level_values(0).unique():
        if len(df.loc[dt]) < 5:
            #'参与计算标的小于5只时，跳过'
            continue
        ic = df['factor_standardized'][dt].rank().corr(df['ret_f'][dt].rank(),method='pearson')
        ic_data[dt] = ic
        
    return ic_data

def get_ic_desc(ic_data):
    """
       因子信息系数的描述
       :param ic_data:{pd.Series，Index[date,]} --rankIC值, 
       :return: {tuple(mean,std,t_static,p_value)}
    """
    tmp = ic_data.dropna()
    mean = tmp.mean()
    std = tmp.std()
    t_static,p_value = st.ttest_ind(tmp, [0] * len(tmp))
    return mean,std,t_static,p_value
    
def get_ic_ir(ic_data):
    return ic_data.mean()/ic_data.std()

def auto_describe(df):
    report = sv.analyze(df,pairwise_analysis='auto')
    report.show_notebook()
    
def get_winning_rate(ic_data_df):
    """
       因子胜率
       :param ic_data_df:{pd.DataFrame，Index[date,]} --rankIC值, 
       :return: {pd.DataFrame}
    """
    ic_mean_sign = np.sign(ic_data_df.mean())
    return (np.sign(ic_data_df) == ic_mean_sign).sum()/ic_data_df.count()
    