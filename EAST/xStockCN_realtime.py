# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2018-2020 azai/Rgveda/GolemQuant
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import datetime
from datetime import datetime as dt, timezone, timedelta, date, time
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pymongo

try:
    import QUANTAXIS as QA
except:
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.fetch.StockCN_realtime modules')
    pass

try:
    from GolemQ.utils.parameter import (
        AKA, 
        INDICATOR_FIELD as FLD, 
        TREND_STATUS as ST
    )
except:
    class AKA():
        """
        常量，专有名称指标，定义成常量可以避免直接打字符串造成的拼写错误。
        """

        # 蜡烛线指标
        CODE = 'code'
        NAME = 'name'
        OPEN = 'open'
        HIGH = 'high'
        LOW = 'low'
        CLOSE = 'close'
        VOLUME = 'volume'
        VOL = 'vol'
        DATETIME = 'datetime'
        LAST_CLOSE = 'last_close'
        PRE_CLOSE = 'pre_close'

        def __setattr__(self, name, value):
            raise Exception(u'Const Class can\'t allow to change property\' value.')
            return super().__setattr__(name, value)

from QUANTAXIS.QAUtil import (
    QASETTING,
    )
client = QASETTING.client['QAREALTIME']
from GolemQ.utils.symbol import (
    normalize_code
)

def GQ_fetch_stock_realtime_adv(code=None,
    num=1,
    collections=client.get_collection('realtime_{}'.format(date.today())),
    verbose=True,
    suffix=False,):
    '''
    返回当日的上下五档, code可以是股票可以是list, num是每个股票获取的数量
    :param code:
    :param num:
    :param collections:  realtime_XXXX-XX-XX 每天实时时间
    :param suffix:  股票代码是否带沪深交易所后缀
    :return: DataFrame
    '''
    if code is not None:
        # code 必须转换成list 去查询数据库，因为五档数据用一个collection保存了股票，指数及基金，所以强制必须使用标准化代码
        if isinstance(code, str):
            code = [normalize_code(code)]
        elif isinstance(code, list):
            code = [normalize_code(symbol) for symbol in code]
            pass
        else:
            print("QA Error GQ_fetch_stock_realtime_adv parameter code is not List type or String type")
        #print(verbose, code)
        items_from_collections = [
            item for item in collections.find({'code': {
                    '$in': code
                }},
                limit=num * len(code),
                sort=[('datetime',
                       pymongo.DESCENDING)])
        ]
        if (items_from_collections is None) or \
            (len(items_from_collections) == 0):
            if verbose:
                print("QA Error GQ_fetch_stock_realtime_adv find parameter code={} num={} collection={} return None"
                    .format(code,
                            num,
                            collections))
            return None
        data = pd.DataFrame(items_from_collections)
        if (suffix == False):
            # 返回代码数据中是否包含交易所代码
            data['code'] = data.apply(lambda x: x.at['code'][:6], axis=1)
        data_set_index = data.set_index(['datetime',
                                         'code'],
                                        drop=False).drop(['_id'],
                                                         axis=1)

        return data_set_index
    else:
        print("QA Error GQ_fetch_stock_realtime_adv parameter code is None")


def GQ_data_tick_resample_1min(tick, type_='1min', if_drop=True, stack_vol=True):
    """
    tick 采样为 分钟数据
    1. 仅使用将 tick 采样为 1 分钟数据
    2. 仅测试过，与通达信 1 分钟数据达成一致
    3. 经测试，可以匹配 QA.QA_fetch_get_stock_transaction 得到的数据，其他类型数据未测试
    demo:
    df = QA.QA_fetch_get_stock_transaction(package='tdx', code='000001',
                                           start='2018-08-01 09:25:00',
                                           end='2018-08-03 15:00:00')
    df_min = QA_data_tick_resample_1min(df)
    """
    tick = tick.assign(amount=tick.price * tick.vol)
    resx = pd.DataFrame()
    _dates = set(tick.date)

    for date in sorted(list(_dates)):
        _data = tick.loc[tick.date == date]
        # morning min bar
        if (stack_vol):
            #_data1 = _data[time(9,
            #                    25):time(11,
            #                             30)].resample(
            #                                 type_,
            #                                 closed='left',
            #                                 offset="30min",
            #                                 loffset=type_
            #                             ).apply(
            #                                 {
            #                                     'price': 'ohlc',
            #                                     'vol': 'sum',
            #                                     'code': 'last',
            #                                     'amount': 'sum'
            #                                 }
            #                             )
            _data1 = _data[time(9,
                                25):time(11,
                                         30)].resample(type_,
                                             closed='left',
                                             offset="30min",).apply({
                                                 'price': 'ohlc',
                                                 'vol': 'sum',
                                                 'code': 'last',
                                                 'amount': 'sum'
                                             })
            _data1.index = _data1.index + to_offset(type_)
        else:
            # 新浪l1快照数据不需要累加成交量 -- 阿财 2020/12/29
            #_data1 = _data[time(9,
            #                    25):time(11,
            #                             30)].resample(
            #                                 type_,
            #                                 closed='left',
            #                                 base=30,
            #                                 loffset=type_
            #                             ).apply(
            #                                 {
            #                                     'price': 'ohlc',
            #                                     'vol': 'last',
            #                                     'code': 'last',
            #                                     'amount': 'last'
            #                                 }
            #                             )
            _data1 = _data[time(9,
                                25):time(11,
                                         30)].resample(type_,
                                             closed='left',
                                             offset="30min",).apply({
                                                 'price': 'ohlc',
                                                 'vol': 'last',
                                                 'code': 'last',
                                                 'amount': 'last'
                                             })
            #print( _data1.index)
            _data1.index = _data1.index + to_offset(type_)
        _data1.columns = _data1.columns.droplevel(0)
        # do fix on the first and last bar
        # 某些股票某些日期没有集合竞价信息，譬如 002468 在 2017 年 6 月 5 日的数据
        if len(_data.loc[time(9, 25):time(9, 25)]) > 0:
            _data1.loc[time(9,
                            31):time(9,
                                     31),
                       'open'] = _data1.loc[time(9,
                                                 26):time(9,
                                                          26),
                                            'open'].values
            _data1.loc[time(9,
                            31):time(9,
                                     31),
                       'high'] = _data1.loc[time(9,
                                                 26):time(9,
                                                          31),
                                            'high'].max()
            _data1.loc[time(9,
                            31):time(9,
                                     31),
                       'low'] = _data1.loc[time(9,
                                                26):time(9,
                                                         31),
                                           'low'].min()
            _data1.loc[time(9,
                            31):time(9,
                                     31),
                       'vol'] = _data1.loc[time(9,
                                                26):time(9,
                                                         31),
                                           'vol'].sum()
            _data1.loc[time(9,
                            31):time(9,
                                     31),
                       'amount'] = _data1.loc[time(9,
                                                   26):time(9,
                                                            31),
                                              'amount'].sum()
        ## 通达信分笔数据有的有 11:30 数据，有的没有
        #if len(_data.loc[time(11, 30):time(11, 30)]) > 0:
        #    _data1.loc[time(11,
        #                    30):time(11,
        #                             30),
        #               'high'] = _data1.loc[time(11,
        #                                         30):time(11,
        #                                                  31),
        #                                    'high'].max()
        #    _data1.loc[time(11,
        #                    30):time(11,
        #                             30),
        #               'low'] = _data1.loc[time(11,
        #                                        30):time(11,
        #                                                 31),
        #                                   'low'].min()
        #    print(len(_data1.loc[time(11,
        #                    30):time(11,
        #                             30),
        #               'close']), _data1.loc[time(11,
        #                    30):time(11,
        #                             30),
        #               'close'], len(_data1.loc[time(11,
        #                                          31):time(11,
        #                                                   31),
        #                                     'close'].values))
        #    _data1.loc[time(11,
        #                    30):time(11,
        #                             30),
        #               'close'] = _data1.loc[time(11,
        #                                          31):time(11,
        #                                                   31),
        #                                     'close'].values
        #    _data1.loc[time(11,
        #                    30):time(11,
        #                             30),
        #               'vol'] = _data1.loc[time(11,
        #                                        30):time(11,
        #                                                 31),
        #                                   'vol'].sum()
        #    _data1.loc[time(11,
        #                    30):time(11,
        #                             30),
        #               'amount'] = _data1.loc[time(11,
        #                                           30):time(11,
        #                                                    31),
        #                                      'amount'].sum()
        _data1 = _data1.loc[time(9, 31):time(11, 30)]

        # afternoon min bar
        if (stack_vol):
            #_data2 = _data[time(13,
            #                    0):time(15,
            #                            0)].resample(
            #                                type_,
            #                                closed='left',
            #                                base=30,
            #                                loffset=type_
            #                            ).apply(
            #                                {
            #                                    'price': 'ohlc',
            #                                    'vol': 'sum',
            #                                    'code': 'last',
            #                                    'amount': 'sum'
            #                                }
            #                            )
            _data2 = _data[time(13,
                                0):time(15,
                                        0)].resample(type_,
                                             closed='left',
                                             offset="30min",).apply({
                                                 'price': 'ohlc',
                                                 'vol': 'sum',
                                                 'code': 'last',
                                                 'amount': 'sum'
                                             })
            _data1.index = _data1.index + to_offset(type_)
        else:
            # 新浪l1快照数据不需要累加成交量 -- 阿财 2020/12/29
            #_data2 = _data[time(13,
            #                    0):time(15,
            #                            0)].resample(
            #                                type_,
            #                                closed='left',
            #                                base=30,
            #                                loffset=type_
            #                            ).apply(
            #                                {
            #                                    'price': 'ohlc',
            #                                    'vol': 'last',
            #                                    'code': 'last',
            #                                    'amount': 'last'
            #                                }
            #                            )
            _data2 = _data[time(13,
                                0):time(15,
                                        0)].resample(type_,
                                             closed='left',
                                             offset="30min",).apply({
                                                 'price': 'ohlc',
                                                 'vol': 'sum',
                                                 'code': 'last',
                                                 'amount': 'sum'
                                             })
            _data1.index = _data1.index + to_offset(type_)

        _data2.columns = _data2.columns.droplevel(0)
        # 沪市股票在 2018-08-20 起，尾盘 3 分钟集合竞价
        if (pd.Timestamp(date) < pd.Timestamp('2018-08-20')) and (tick.code.iloc[0][0] == '6'):
            # 避免出现 tick 数据没有 1:00 的值
            if len(_data.loc[time(13, 0):time(13, 0)]) > 0:
                _data2.loc[time(15,
                                0):time(15,
                                        0),
                           'high'] = _data2.loc[time(15,
                                                     0):time(15,
                                                             1),
                                                'high'].max()
                _data2.loc[time(15,
                                0):time(15,
                                        0),
                           'low'] = _data2.loc[time(15,
                                                    0):time(15,
                                                            1),
                                               'low'].min()
                _data2.loc[time(15,
                                0):time(15,
                                        0),
                           'close'] = _data2.loc[time(15,
                                                      1):time(15,
                                                              1),
                                                 'close'].values
        else:
            # 避免出现 tick 数据没有 15:00 的值
            if len(_data.loc[time(13, 0):time(13, 0)]) > 0:
                if (len(_data2.loc[time(15, 1):time(15, 1)]) > 0):
                    _data2.loc[time(15,
                                    0):time(15,
                                            0)] = _data2.loc[time(15,
                                                                  1):time(15,
                                                                          1)].values
                else:
                    # 这种情况下每天下午收盘后15:00已经具有tick值，不需要另行额外填充
                    #  -- 阿财 2020/05/27
                    #print(_data2.loc[time(15,
                    #                0):time(15,
                    #                        0)])
                    pass
        _data2 = _data2.loc[time(13, 1):time(15, 0)]
        resx = resx.append(_data1).append(_data2)
    resx['vol'] = resx['vol'] * 100.0
    resx['volume'] = resx['vol']
    resx['type'] = '1min'
    if if_drop:
        resx = resx.dropna()
    return resx.reset_index().drop_duplicates().set_index(['datetime', 'code'])


def GQ_fetch_stock_day_realtime_adv(codelist, 
                                    data_day, 
                                    verbose=True):
    """
    查询日线实盘数据，支持多股查询
    """
    if codelist is not None:
        # codelist 必须转换成list 去查询数据库
        if isinstance(codelist, str):
            codelist = [codelist]
        elif isinstance(codelist, list):
            pass
        else:
            print("QA Error GQ_fetch_stock_day_realtime_adv parameter codelist is not List type or String type")
    start_time = dt.strptime(str(dt.now().date()) + ' 09:15', '%Y-%m-%d %H:%M')
    if (len(data_day.data.index.get_level_values(level=0)) == 0):
        print(u'K线数据长度为零：', codelist)
    elif ((dt.now() > start_time) and ((dt.now() - data_day.data.index.get_level_values(level=0)[-1].to_pydatetime()) > timedelta(hours=10))) or \
        ((dt.now() < start_time) and ((dt.now() - data_day.data.index.get_level_values(level=0)[-1].to_pydatetime()) > timedelta(hours=40))):
        if (verbose == True):
            print('时间戳差距超过：', dt.now() - data_day.data.index.get_level_values(level=0)[-1].to_pydatetime(),
                  '尝试查找日线实盘数据....', codelist)
        
        try:
            if (dt.now() > start_time):
                collections = client.get_collection('realtime_{}'.format(date.today()))
            else:
                collections = client.get_collection('realtime_{}'.format(date.today() - timedelta(hours=24)))
            data_realtime = GQ_fetch_stock_realtime_adv(codelist, num=8000, 
                                                        verbose=verbose, suffix=False,
                                                        collections=collections)
        except: 
            data_realtime = GQ_data_tick_resample_1min(codelist, num=8000, verbose=verbose)
        if (data_realtime is not None) and \
            (len(data_realtime) > 0):
            # 合并实盘实时数据
            data_realtime = data_realtime.drop_duplicates((["datetime",
                        'code'])).set_index(["datetime",
                        'code'],
                            drop=False)
            data_realtime = data_realtime.reset_index(level=[1], drop=True)
            data_realtime['date'] = pd.to_datetime(data_realtime['datetime']).dt.strftime('%Y-%m-%d')
            data_realtime['datetime'] = pd.to_datetime(data_realtime['datetime'])
            for code in codelist:
                # 顺便检查股票行情长度，发现低于30天直接砍掉。
                if (len(data_day.select_code(code[:6])) < 30):
                    print(u'{} 行情只有{}天数据，新股或者数据不足，不进行择时分析。'.format(code, 
                                                                   len(data_day.select_code(code[:6]))))
                    try:
                        data_day.data.drop(data_day.select_code(code).data, 
                                           inplace=True)
                    except:
                        pass
                    continue

                # *** 注意，QA_data_tick_resample_1min 函数不支持多标的 *** 需要循环处理
                data_realtime_code = data_realtime[data_realtime['code'].eq(code[:6])]
                if (len(data_realtime_code) > 0):
                    data_realtime_code = data_realtime_code.set_index(['datetime']).sort_index()
                    if ('volume' in data_realtime_code.columns) and \
                        ('vol' not in data_realtime_code.columns):
                        # 我也不知道为什么要这样转来转去，但是各家(新浪，pytdx)l1数据就是那么不统一
                        data_realtime_code.rename(columns={"volume": "vol"}, 
                                                    inplace = True)
                    elif ('volume' in data_realtime_code.columns):
                        data_realtime_code['vol'] = np.where(np.isnan(data_realtime_code['vol']), 
                                                             data_realtime_code['volume'], 
                                                             data_realtime_code['vol'])

                    # 一分钟数据转出来了
                    try:
                        data_realtime_1min = GQ_data_tick_resample_1min(data_realtime_code, 
                                                                        type_='1min',
                                                                        stack_vol=False)
                        #data_realtime_1min['vol']
                    except:
                        print('fooo1', code)
                        print(data_realtime_code)
                        raise('foooo1{}'.format(code))
                    data_realtime_1day = QA.QA_data_min_to_day(data_realtime_1min)
                    if (len(data_realtime_1day) > 0):
                        # 转成日线数据
                        data_realtime_1day.rename(columns={"vol": "volume"}, 
                                                  inplace = True)

                        # 假装复了权，我建议复权那几天直接量化处理，复权几天内对策略买卖点影响很大
                        data_realtime_1day['adj'] = 1.0 
                        data_realtime_1day['datetime'] = pd.to_datetime(data_realtime_1day.index)
                        data_realtime_1day = data_realtime_1day.set_index(['datetime', 'code'], 
                                                                        drop=True).sort_index()
                        
                        # 当早盘集合竞价未出现成交，9:30分的Open和Low报价会是0元，特别处理
                        pre_close = data_day.data[AKA.CLOSE].tail(1).item()
                        if (data_realtime_1day[AKA.OPEN].head(1).item() < 0.001):
                            data_realtime_1day.loc[data_realtime_1day.index.get_level_values(level=0)[0],
                                                   AKA.OPEN] = pre_close
                            data_realtime_1day.loc[data_realtime_1day.index.get_level_values(level=0)[0],
                                                   AKA.LOW] = min(data_realtime_1day[AKA.OPEN].head(1).item(), 
                                                                  data_realtime_1day[AKA.HIGH].head(1).item(), 
                                                                  data_realtime_1day[AKA.CLOSE].head(1).item())

                        if (data_day.data.index.get_level_values(level=0)[-1] != data_realtime_1day.index.get_level_values(level=0)[-1]):
                            # 成功获取到 l1 实盘数据，获取主力资金流向
                            if (data_realtime_1day.index.get_level_values(level=0)[-1] > dt.now()):
                                print(u'尝试追加资金流向数据，股票代码：{} 时间：{} 价格：{}'.format(data_realtime_1day.index[0][1],
                                                                                             data_realtime_1day.index[-1][0],
                                                                                             data_realtime_1day[AKA.CLOSE][-1]))

                            if (verbose == True):
                                print(u'追加实时实盘数据 {}，股票代码：{} 时间：{} 价格：{}'.format(len(data_realtime_1day), 
                                                                                             data_realtime_1day.index[0][1],
                                                                                             data_realtime_1day.index[-1][0],
                                                                                             data_realtime_1day[AKA.CLOSE][-1]))
                            data_day.data = data_day.data.append(data_realtime_1day, 
                                                                 sort=True)

    return data_day


def GQ_fetch_stock_min_realtime_adv(codelist,
                                    data_min,
                                    frequency, 
                                    verbose=True):
    """
    查询A股的指定小时/分钟线线实盘数据
    """
    if codelist is not None:
        # codelist 必须转换成list 去查询数据库
        if isinstance(codelist, str):
            codelist = [codelist]
        elif isinstance(codelist, list):
            pass
        else:
            if verbose:
                print("QA Error GQ_fetch_stock_min_realtime_adv parameter codelist is not List type or String type")

    if data_min is None:
        if verbose:
            print(u'代码：{} 今天停牌或者已经退市*'.format(codelist))  
        return None

    try:
        foo = (dt.now() - data_min.data.index.get_level_values(level=0)[-1].to_pydatetime())
    except:
        if verbose:
            print(u'代码：{} 今天停牌或者已经退市**'.format(codelist))                    
        return None
    start_time = dt.strptime(str(dt.now().date()) + ' 09:15', '%Y-%m-%d %H:%M')
    if ((dt.now() > start_time) and ((dt.now() - data_min.data.index.get_level_values(level=0)[-1].to_pydatetime()) > timedelta(hours=10))) or \
        ((dt.now() < start_time) and ((dt.now() - data_min.data.index.get_level_values(level=0)[-1].to_pydatetime()) > timedelta(hours=24))):
        if (verbose == True):
            print('时间戳差距超过：', dt.now() - data_min.data.index.get_level_values(level=0)[-1].to_pydatetime(),
                  '尝试查找分钟线实盘数据....', codelist)

        if (dt.now() > start_time):
            collections = client.get_collection('realtime_{}'.format(date.today()))
        else:
            collections = client.get_collection('realtime_{}'.format(date.today() - timedelta(hours=24)))
        for code in codelist:
            #print(u'查询实盘数据。', code, )
            try:
                data_realtime = GQ_fetch_stock_realtime_adv(code, num=8000, 
                                                            verbose=verbose, suffix=False, 
                                                            collections=collections)
            except: 
                data_realtime = QA.QA_fetch_stock_realtime_adv(code, num=8000, verbose=verbose)

            if (data_realtime is not None) and \
                (len(data_realtime) > 0):
                # 合并实盘实时数据
                data_realtime = data_realtime.drop_duplicates((["datetime",
                            'code'])).set_index(["datetime",
                            'code'],
                                drop=False)

                data_realtime = data_realtime.reset_index(level=[1], drop=True)
                data_realtime['date'] = pd.to_datetime(data_realtime['datetime']).dt.strftime('%Y-%m-%d')
                data_realtime['datetime'] = pd.to_datetime(data_realtime['datetime'])

                # 顺便检查股票行情长度，发现低于30天直接砍掉。
                try:
                    if (len(data_min.select_code(code[:6])) < 30):
                        if verbose:
                            print(u'{} 行情只有{}天数据，新股或者数据不足，不进行择时分析。新股买不买卖不卖，建议掷骰子。'.format(code, 
                                                                       len(data_min.select_code(code))))
                        data_min.data.drop(data_min.select_code(code), inplace=True)
                        continue
                except:
                    print('Error!')
                    if verbose:
                        print(u'代码：{} 今天停牌或者已经退市***'.format(code))                    
                    continue

                # *** 注意，QA_data_tick_resample_1min 函数不支持多标的 *** 需要循环处理
                # 可能出现8位六位股票代码兼容问题
                data_realtime_code = data_realtime[data_realtime['code'].eq(code[:6])]

                if (len(data_realtime_code) > 0):
                    data_realtime_code = data_realtime_code.set_index(['datetime']).sort_index()
                    if ('volume' in data_realtime_code.columns) and \
                        ('vol' not in data_realtime_code.columns):
                        # 我也不知道为什么要这样转来转去，但是各家(新浪，pytdx)l1数据就是那么不统一
                        data_realtime_code.rename(columns={"volume": "vol"}, 
                                                  inplace = True)
                    elif ('volume' in data_realtime_code.columns):
                        data_realtime_code['vol'] = np.where(np.isnan(data_realtime_code['vol']), 
                                                             data_realtime_code['volume'], 
                                                             data_realtime_code['vol'])

                    # 将l1 Tick数据重采样为1分钟
                    try:
                        data_realtime_1min = GQ_data_tick_resample_1min(data_realtime_code, 
                                                                        type_='1min',
                                                                        stack_vol=False)
                    except:
                        if verbose:
                            print('fooo1', code)
                            print(data_realtime_code)
                        pass
                        #raise('foooo1{}'.format(code))

                    if (len(data_realtime_1min) == 0):
                        # 没有数据或者数据缺失，尝试获取腾讯财经的1分钟数据
                        #import easyquotation
                        #quotation = easyquotation.use("timekline")
                        #data = quotation.real(codelist, prefix=False)
                        #if verbose:
                        #    print(data)
                        pass
                        return data_min

                    # 一分钟数据转出来了，重采样为指定小时/分钟线数据
                    data_realtime_1min = data_realtime_1min.reset_index([1], drop=False)
                    data_realtime_mins = QA.QA_data_min_resample(data_realtime_1min, 
                                                                 type_=frequency)

                    if (len(data_realtime_mins) > 0):
                        # 转成指定分钟线数据
                        data_realtime_mins.rename(columns={"vol": "volume"}, 
                                                  inplace = True)

                        # 假装复了权，我建议复权那几天直接量化处理，复权几天内对策略买卖点影响很大
                        data_realtime_mins['adj'] = 1.0 
                        #data_realtime_mins['datetime'] =
                        #pd.to_datetime(data_realtime_mins.index)
                        #data_realtime_mins =
                        #data_realtime_mins.set_index(['datetime', 'code'],
                        #                                                drop=True).sort_index()

                        # 当早盘集合竞价未出现成交，9:30分的Open和Low报价会是0元，特别处理
                        pre_close = data_min.data[AKA.CLOSE].tail(1).item()
                        if (data_realtime_mins[AKA.OPEN].head(1).item() < 0.001):
                            data_realtime_mins.loc[data_realtime_mins.index.get_level_values(level=0)[0],
                                                   AKA.OPEN] = pre_close
                            data_realtime_mins.loc[data_realtime_mins.index.get_level_values(level=0)[0],
                                                   AKA.LOW] = min(data_realtime_1min[AKA.LOW][1:].min(), 
                                                                  data_realtime_1min[AKA.HIGH][1:].min(), 
                                                                  data_realtime_1min[AKA.CLOSE][1:].min(),)
                            #print(u'追加实时实盘数据，股票代码：{} 时间：{} 开盘
                            #{}：价格：{}'.format(data_realtime_mins.index[0][1],
                            #                                                         data_realtime_mins.index[-1][0],
                            #                                                         data_realtime_mins[AKA.OPEN][0],
                            #                                                         data_realtime_mins[AKA.CLOSE][-1]))

                  #      if (len(data_realtime_mins) > 0):
                  #          print(u'分钟线 status:', (dt.now() < start_time),
                  #          '时间戳差距超过：', dt.now() -
                  #          data_min.data.index.get_level_values(level=0)[-1].to_pydatetime(),
                  #'尝试查找实盘数据....', codelist)
                  #          print(data_min.data.tail(3), data_realtime_mins)
                        if (data_min.select_code(code[:6]).index.get_level_values(level=0)[-1] != data_realtime_mins.index.get_level_values(level=0)[-1]):
                            if (verbose == True):
                                #print(data_min.data.tail(3),
                                #data_realtime_mins)
                                print(u'追加实时实盘数据 {}，股票代码：{}({}) 开盘 {}：价格：{}'.format(len(data_realtime_mins),
                                                                                     code, data_realtime_mins.index[0][1],
                                                                                     data_realtime_mins.index[-1][0],
                                                                                     data_realtime_mins[AKA.OPEN][0],
                                                                                     data_realtime_mins[AKA.CLOSE][-1]))
                            data_min.data = data_min.data.append(data_realtime_mins, 
                                                                 sort=True)

                        # Amount, Volume 计算不对
    else:
        if (verbose == True):
            print(u'没有时间差', dt.now() - data_min.data.index.get_level_values(level=0)[-1].to_pydatetime())
    data_min.data = data_min.data.sort_index()
    return data_min

    #
    #data_realtime_1min = data_realtime_1min.reset_index(level=[1], drop=False)

    #data_realtime_5min = QA.QA_data_min_resample(data_realtime_1min,
    #                                             type_='5min')
    #print(data_realtime_5min)

    #data_realtime_15min = QA.QA_data_min_resample(data_realtime_1min,
    #                                              type_='15min')
    #print(data_realtime_15min)

    #data_realtime_30min = QA.QA_data_min_resample(data_realtime_1min,
    #                                              type_='30min')
    #print(data_realtime_30min)
    #data_realtime_1hour = QA.QA_data_min_resample(data_realtime_1min,
    #                                             type_='60min')
    #print(data_realtime_1hour)
    #return data_min
def GQ_fetch_index_min_realtime_adv(codelist,
                                    data_min,
                                    frequency, 
                                    verbose=True):
    """
    查询指数和ETF的分钟线实盘数据
    """
    # 将l1 Tick数据重采样为1分钟
    data_realtime_1min = data_realtime_1min.reset_index(level=[1], drop=False)

    # 检查 1min数据是否完整，如果不完整，需要从腾讯财经获取1min K线
    #if ():


    data_realtime_5min = QA.QA_data_min_resample(data_realtime_1min, 
                                                 type_='5min')
    print(data_realtime_5min)

    data_realtime_15min = QA.QA_data_min_resample(data_realtime_1min, 
                                                  type_='15min')
    print(data_realtime_15min)

    data_realtime_30min = QA.QA_data_min_resample(data_realtime_1min, 
                                                  type_='30min')
    print(data_realtime_30min)
    data_realtime_1hour = QA.QA_data_min_resample(data_realtime_1min,
                                                 type_='60min')
    print(data_realtime_1hour)
    return data_min


if __name__ == '__main__':
    """
    用法示范
    """
    codelist = ['600157', '300263']
    data_min = QA.QA_fetch_stock_min_adv(codelist,
                                        '2008-01-01',
                                        '{}'.format(date.today(),),
                                        frequence='15min')

    data_min = GQ_fetch_stock_min_realtime_adv(codelist, data_min,
                                               frequency='15min')