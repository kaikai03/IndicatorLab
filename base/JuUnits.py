
from IPython.display import display
import inspect

# from memory_profiler import profile 
# @profile

# from tqdm.autonotebook import tqdm

######移动文件后如果无法访问，加上这一句
# import sys
# import os 
# module_path = os.path.abspath(os.path.join('..')) 
# if module_path not in sys.path: 
#     sys.path.append(module_path)
    
from tqdm.notebook import tqdm
from multiprocessing import Pool,freeze_support,cpu_count
from functools import partial
from importlib import import_module
import os
from numpy.lib.stride_tricks import as_strided as stride
import pandas as pd
import numpy as np
from random import randint
import pymongo

from datetime import (
    date as da,
    datetime as dt,
    timezone, timedelta
)

from QUANTAXIS.QAUtil import (
    DATABASE,
    QA_util_get_next_day,
    QA_util_get_real_date,
    QA_util_log_info,
    QA_util_to_json_from_pandas,
    trade_date_sse
)

from QUANTAXIS.QAUtil import (
    QA_util_code_tolist,
    QA_util_time_stamp,
    QA_util_date_valid,
    QA_util_date_stamp
)

import time
from datetime import (
    date,
    datetime as dt,
    timezone, timedelta
)






def parallal_task(worker, func, iterable, **kwargs): 
    with open('../multi_fun_box/{}.py'.format(func.__name__), 'w', encoding="utf-8") as file:
        file.write(inspect.getsource(func)) 
    
    module = import_module("multi_fun_box.{}".format(func.__name__))
    task = getattr(module, func.__name__)

    freeze_support()
    '子进程PID：', os.getpid(), '主进程PPID', os.getppid()
    print('Now in the main code. Process name is:', __name__)
    print(('%s, subpid:%d  pid:%d') % (__name__, os.getpid(),os.getppid()))
    
    if worker == None or worker ==0:
        worker = cpu_count()
        
    p=Pool(processes = worker)
    func_ = partial(task,**kwargs)
#     res = list(tqdm(p.imap_unordered(func_, iterable), total=len(iterable)))
    res = list(tqdm(p.imap(func_, iterable), total=len(iterable)))
#     res = list(p.imap_unordered(func_, iterable))
    p.close()
    p.join()
    return res

def task_chunk_split(array, n):
    assert n > 0, ('split to %d chunk ????????') % n
    assert n <= len(array), 'chunk > len(array)????????'
    chunk_size = [0]*n
    
    for i,item in enumerate(array):
        chunk_size[i%n] +=1

    cum_count=0
    tmp = []
    for si in chunk_size:
        tmp.append(array[cum_count:cum_count+si])
        cum_count += si
    return tmp


display_handle = None
def stream_print_on():
    global display_handle
    display_handle = display("display",display_id=True)

def stream_print(str):
    assert display_handle!=None,"需要先stream_print_on,或使用装饰器"
    display_handle.update(str)

def stream_print_off():
    global display_handle
    display_handle = None
    
def stream_print_wrap(f):
    def inner(*args,**kwargs):
        stream_print_on()
        ret =f(*args,**kwargs)
        stream_print_off()
        return ret
    return inner

def help_source_code(moulde):
    return inspect.getsourcelines(moulde)

def excute_for_multidates(data, func, level=0, **pramas):
    return data.groupby(level=level, group_keys=False).apply(func,**pramas)

def roll_multi_result(df: pd.DataFrame, apply_func: callable, window: int, return_col_num: int, **kwargs):
    """
    rolling with multiple columns on 2 dim pd.Dataframe
    * the result can apply the function which can return pd.Series with multiple columns

    call apply function with numpy ndarray
    
    :param return_col_num: apply_func返回的个数（列数）
    :param apply_func: --注意：传递的参数，前N个为原index，N=index的维数
    :param df: [pd.DataFrame,pd.Series]
    :param window: 滚动窗口
    :param kwargs: 
    :return:
    """

    # move index to values
    v = df.reset_index().values

    dim0, dim1 = v.shape
    stride0, stride1 = v.strides

    stride_values = stride(v, (dim0 - (window - 1), window, dim1), (stride0, stride0, stride1))

    result_values = np.full((dim0, return_col_num), np.nan)

    for idx, values in enumerate(stride_values, window - 1):
        # values : col 1 is index, other is value
        result_values[idx,] = apply_func(values, **kwargs)

    return result_values



def fetch_index_day_common(field_name, field_value, start, end,collections,  format='pd'):
    '通用数据查询'
    start = str(start)[0:10]
    end = str(end)[0:10]
    
    field_value_ = field_value
    if not isinstance(field_value_, list):
        field_value_ = [field_value_]
        
    cursor = collections.find(
        {
            field_name: {
                '$in': field_value_
            },
            "date_stamp":
                {
                    "$lte": QA_util_date_stamp(end),
                    "$gte": QA_util_date_stamp(start)
                }
        },
        {"_id": 0},
        batch_size=10000
    )

    res = pd.DataFrame([item for item in cursor])
    try:
        res = res.assign(
            date=pd.to_datetime(res.date, utc=False)
        ).set_index('date',drop=True)
    except:
        res = None

    if format in ['P', 'p', 'pandas', 'pd']:
        return res
    elif format in ['json', 'dict']:
        return QA_util_to_json_from_pandas(res)
    # 多种数据格式
    elif format in ['n', 'N', 'numpy']:
        return numpy.asarray(res)
    elif format in ['list', 'l', 'L']:
        return numpy.asarray(res).tolist()
    else:
        print(
            "Error fetch_index_day_em format parameter %s is none of  \"P, p, pandas, pd , n, N, numpy !\" "
            % format
        )
        return None
    
    
def now_time(separate_hour=15):
    return str(da.today() - timedelta(days=1)) + \
            ' 17:00:00' if dt.now().hour < separate_hour else  str(da.today())+' 15:00:00'

def now_time_tradedate():
    return str(QA_util_get_real_date(str(da.today() - timedelta(days=1)), trade_date_sse, -1)) + \
           ' 17:00:00' if dt.now().hour < 15 else str(QA_util_get_real_date(
        str(da.today()), trade_date_sse, -1)) + ' 15:00:00'

def date_range(begin_date,end_date):
    date_list = []
    begin_date = dt.strptime(begin_date, "%Y-%m-%d")
    end_date = dt.strptime(end_date,"%Y-%m-%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        begin_date += timedelta(days=1)
    return date_list


def get_root_path(project_name='IndicatorLab'):
    """
    获取当前项目根路径
    :param project_name: 项目名称
    :return: 指定项目的根路径
    """
    p_name = 'IndicatorLab' if project_name is None else project_name
    project_path = os.path.abspath(os.path.dirname(__file__))
    # Windows
    if project_path.find('\\') != -1: separator = '\\'
    # Mac、Linux、Unix
    if project_path.find('/') != -1: separator = '/'
    root_path = project_path[:project_path.find(f'{p_name}{separator}') + len(f'{p_name}{separator}')]
    return root_path
