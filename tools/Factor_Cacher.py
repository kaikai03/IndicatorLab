
import sys
import os 
module_path = os.path.abspath(os.path.join('..')) 
if module_path not in sys.path: 
    sys.path.append(module_path)

import warnings
import pandas as pd
import numpy as np

import tools.Sample_Tools as smpl
import base.JuUnits as ju


__cache_dir__ = ju.get_root_path()+'/cache.feather/'
__cache_type__ = '.feather'

# df = smpl.get_sample_by_zs(name='沪深300', end='2021-11-28', gap=2500,  only_main=True, filter_st=True).data
# df.to_parquet(__cache_dir__+'test.parquet')

def file_path(file_name):
    return __cache_dir__ + file_name + __cache_type__

def is_cache_exist(cache_name):
    return os.path.exists(file_path(cache_name))

def save_cache(name, data):
    data.reset_index().to_feather(file_path(name))
    
def load_cache(cache_name:str, to_series:bool=False):
    assert is_cache_exist(cache_name),'cache not exist, create first'
    df = pd.read_feather(file_path(cache_name)).set_index(['date','code'])
    if to_series:
        df =  df.squeeze()
    return df

def load_cache_adv(cache_name:str, start:str, end:str, to_series:bool=False, flag_time:str='date'):
    assert is_cache_exist(cache_name),'cache not exist, create first'
        
    st = pd.Timestamp(start) 
    en = pd.Timestamp(end) 

    df = pd.read_feather(file_path(cache_name)).set_index([flag_time,'code'])[st:en]
    date_idx = df.index.get_level_values(0).unique()
    if not (st >= date_idx.min() and en <= date_idx.max()):
        info = '"{}"：date range out of cache[{},{}]'.format(
            cache_name,
            date_idx.min().strftime('%Y-%m-%d %H:%M:%S'), 
            date_idx.max().strftime('%Y-%m-%d %H:%M:%S')
        )
        warnings.warn(info)
        
    df = df[st:en]
    
    if to_series:
        df =  df.squeeze()
    return df


def load_caches_adv(cache_names:list, start:str=None, end:str=None, flag_time:str='date'):
    if not start is None:
        assert not end is None, 'start and end must not be None at the same time,or be None at the same time'
    
    tmp = []
    for name in cache_names:
        if start is None:
            df = load_cache(name)
        else:
            df = load_cache_adv(name,start,end,flag_time=flag_time)
        tmp.append(df)  
        
    return pd.concat(tmp,axis=1)
     
    
        
