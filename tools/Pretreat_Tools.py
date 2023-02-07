
import sys
import os 
module_path = os.path.abspath(os.path.join('..')) 
if module_path not in sys.path: 
    sys.path.append(module_path)
    
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as st
import statsmodels.api as sm
import tools.Sample_Tools as smpl

# import cpuinfo
# if 'ntel' in cpuinfo.get_cpu_info()['brand_raw']:
# from sklearnex import patch_sklearn, unpatch_sklearn
# unpatch_sklearn() ##注意，少量数据的线性回归没有优势。慎用，存在内存泄露

from sklearn import linear_model

def neutralize(factor:pd.Series, data, categorical:list=None, logarithmetics:list=None):
    '''中性化：
        :param categorical：{list} --指明需要被dummy的列
        :param logarithmetics：{list}  --指明要对对数化的列
        注：被categorical的column的value必须是字符串。
        注：一般来说，顺序是 去极值->中性化->标准化
        注：单截面操作
    '''
    if factor.index.is_monotonic_increasing == False or data.index.is_monotonic_increasing == False:
        import warnings
        warnings.warn('factor or data should be sorted, 否则有可能会造成会自变量和因变量匹配错误',UserWarning)
        
    X = data.copy()
    # 对数化
    if not logarithmetics is None:
        X[logarithmetics] = X[logarithmetics].agg('log')
    # 哑变量
    if not categorical is None:
        X = pd.get_dummies(X,columns=categorical)
        
#     print(X)
        
    model = linear_model.LinearRegression(fit_intercept=False).fit(X, factor)
    neutralize_factor = factor - model.predict(X)

    return neutralize_factor

    

def winsorize_by_quantile(obj, floor=0.025, upper=0.975, column=None, drop=True):
    """
       根据分位上下限选取数据
       :param obj:{pd.DataFrame | pd.Series} 
       :param column:{str} --当obj为DataFrame时，用来指明处理的列。
       :param drop:{bool} --分位外的数据处理方式，
                            True：删除整（行）条数据；
                            False：用临界值替换范围外的值
    """
    if isinstance(obj, pd.Series):
        qt = obj.quantile([floor,upper])
        if drop:
            return obj[(obj>=qt[floor]) & (obj<=qt[upper])]
        else:
            obj[obj < qt[floor]] = qt[floor]
            obj[obj > qt[upper]] = qt[upper]
            return obj
    
    if isinstance(obj, pd.DataFrame):
        assert column, 'COLUMN CANT be NONE when obj is dataframe'
        qt = obj[column].quantile([floor,upper])
        if drop:
            return obj[(obj[column]>=qt[floor]) & (obj[column]<=qt[upper])]
        else:
            obj.loc[obj[column] < qt[floor], column] = qt[floor]
            obj.loc[obj[column] > qt[upper], column] = qt[upper]
            return obj
    raise TypeError('obj must be series or dataframe')
    
def winsorize_by_mad(obj, n=3, column=None, drop=True):
    """
       根据中位数偏离倍数选取数据
       :param obj:{pd.DataFrame | pd.Series} 
       :param n:{pd.DataFrame | pd.Series} --偏离倍数
       :param column:{str} --当obj为DataFrame时，用来指明处理的列。
       :param drop:{bool} --分位外的数据处理方式，
                            True：删除整（行）条数据；
                            False：用临界值替换范围外的值
    """
    
    if isinstance(obj, pd.Series):
        median = np.median(obj.dropna())
        mad = np.median((obj.dropna() - median).abs())
        #样本标准差的估计量(σ≈1.483)
        mad_e = 1.483*mad
        upper = median + n*mad_e
        floor = median - n*mad_e
        if drop:
            return obj[(obj>=floor) & (obj<=upper) | obj.isna()]
        else:
            obj[obj < floor] = floor
            obj[obj > upper] = upper
            return obj
    
    if isinstance(obj, pd.DataFrame):
        assert column, 'COLUMN CANT be NONE when obj is dataframe'
        median = np.median(obj[column].dropna())
        mad = np.median((obj.dropna() - median).abs())
        mad_e = 1.483*mad
        upper = median + n*mad_e
        floor = median - n*mad_e
        if drop:
            return obj[(obj[column]>=floor) & (obj[column]<=upper) | obj[column].isna()]
        else:
            obj.loc[obj[column] < floor, column] = floor
            obj.loc[obj[column] > upper, column] = upper
            return obj
    
    raise TypeError('obj must be series or dataframe')

# 标准化
def standardize(data, multi_code=False):
    if multi_code:
        return data.groupby(level=1, group_keys=False).apply(lambda x: standardize(x,multi_code=False))
    else:
        return (data - data.mean())/data.std()

def binning(df, deal_column:str,box_count:int, labels=None, inplace=True):
    """
       分箱，为df增加名为"group_label"的列作为分组标签。
       :param df:{pd.DataFrame} 
       :param deal_column:{str} --要处理的列名,
       :param box_count:{int} --分几组,
       :param labels:{list} --分组的标签名，默认是分组序号（default:None）
                              默认情况下，生成的标签是反序的，既最小的值在最后的组
       :param inplace:{bool} --是否在原对象上修改,建议用true，效率高（default:True）
       :return: {pd.DataFame}
    """
    assert isinstance(df, pd.DataFrame), 'df必须为dataframe'
    if not labels is None:
        assert len(labels)==box_count, 'labels的数量必须与分箱数相等'
        labels_= labels
    else:
        labels_= np.array(range(box_count))+1
        labels_ = labels_[::-1]
    
    vals = df[deal_column]
    val_set = vals.unique()
    reality_count = len(val_set)
    
    if inplace:
        if box_count > reality_count:
            # 可能由于大量0或者nan，导致分类的数量少于分箱数量。 直接当任务失败，返回空值
            df['group_label'] = None
            return df
        else:
            vals = df[deal_column]
            val_set = vals.unique()
            bins = pd.qcut(val_set, box_count, labels=labels_, retbins=False,)
            val_bin_dic = {key:bin_val for key,bin_val in zip(val_set,bins)}
            res = list(map(lambda x: val_bin_dic[x], vals))
            
            df['group_label'] = res
            return df
    else:
        if box_count > reality_count:
            # 可能由于大量0或者nan，导致分类的数量少于分箱数量。 直接当任务失败，返回空值
            return df.assign(group_label=None)
        else:
            bins = pd.qcut(val_set, box_count, labels=labels_, retbins=False,)
            val_bin_dic = {key:bin_val for key,bin_val in zip(val_set,bins)}
            res = list(map(lambda x: val_bin_dic[x], vals))
            return df.assign(group_label=res)

