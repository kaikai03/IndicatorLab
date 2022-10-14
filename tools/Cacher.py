
import sys
import os 
module_path = os.path.abspath(os.path.join('..')) 
if module_path not in sys.path: 
    sys.path.append(module_path)

from enum import Enum
import warnings
import pandas as pd
import numpy as np

import tools.Sample_Tools as smpl
import base.JuUnits as ju


# __cache_dir__ = ju.get_root_path()+'/cache.feather/'
__cache_dir__ = 'C://cache.feather/'
__file_type__ = '.feather'

class CACHE_TYPE(Enum):
    default = ''
    STOCK = '/stock/'
    FATOR = '/fator/'
    TMP = '/tmp/'
    
# df = smpl.get_sample_by_zs(name='沪深300', end='2021-11-28', gap=2500,  only_main=True, filter_st=True).data
# df.to_parquet(__cache_dir__+'test.parquet')

def file_path(file_name, cache_type=CACHE_TYPE.default):
    return __cache_dir__ + cache_type.value + file_name + __file_type__

def is_cache_exist(cache_name, cache_type=CACHE_TYPE.default):
    return os.path.exists(file_path(cache_name,cache_type))

def save_cache(name, data, cache_type=CACHE_TYPE.default):
    data.reset_index().to_feather(file_path(name,cache_type=cache_type))
    
def load_cache(cache_name:str, to_series:bool=False, time_flag:str='date',cache_type=CACHE_TYPE.default):
    assert is_cache_exist(cache_name,cache_type),'cache not exist, create first'
    df = pd.read_feather(file_path(cache_name, cache_type)).set_index([time_flag,'code']).sort_index(level=0)
    if to_series:
        df =  df.squeeze()
    return df

def load_cache_adv(cache_name:str, start:str, end:str, to_series:bool=False, time_flag:str='date',cache_type=CACHE_TYPE.default):
    assert is_cache_exist(cache_name, cache_type=cache_type),'cache not exist, create first'
        
    st = pd.Timestamp(start) 
    en = pd.Timestamp(end) 

    df = pd.read_feather(file_path(cache_name, cache_type)).set_index([time_flag,'code']).sort_index(level=0)
    dt_index = df.index.get_level_values(0)
    date_range = dt_index.unique()
    if not (st >= date_range.min() and en <= date_range.max()):
        info = '"{}"：date range out of cache[{},{}]'.format(
            cache_name,
            date_range.min().strftime('%Y-%m-%d %H:%M:%S'), 
            date_range.max().strftime('%Y-%m-%d %H:%M:%S')
        )
        warnings.warn(info)
        
    df = df.loc[(dt_index >= start) & (dt_index <= end)]

    if to_series:
        df =  df.squeeze()
    return df


def load_caches_adv(cache_names:list, start:str=None, end:str=None, time_flag:str='date',cache_type=CACHE_TYPE.default):
    if not start is None:
        assert not end is None, 'start and end must not be None at the same time,or be None at the same time'
    assert isinstance(cache_names,list), 'cache_names MUST be list'
    
    tmp = []
    for name in cache_names:
        if start is None:
            df = load_cache(name,cache_type=cache_type)
        else:
            df = load_cache_adv(name,start,end,time_flag=time_flag,cache_type=cache_type)
        tmp.append(df)  
        
    return pd.concat(tmp,axis=1).sort_index()
     
    
        
