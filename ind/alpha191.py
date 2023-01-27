
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing
from sklearn import linear_model

def STD(data, windows):
    return data.rolling(window=windows, min_periods=windows).std()
def MEAN(data, windows):
    return data.rolling(window=windows, min_periods=windows).mean()
def DELTA(data, windows):
    return data.diff(4)
def SEQUENCE(n):
    return pd.Series(np.arange(1,n+1))

def SMA(data,windows,alpha):
    return data.ewm(adjust=False, alpha=float(alpha)/windows, min_periods=windows, ignore_na=False).mean()

def REGBETA(xs, y, n):
    assert len(y)>=n,  'len(y)!>=n !!!'+ str(y.index[0])
    regress = linear_model.LinearRegression(fit_intercept=False)
    def reg(X,Y):
        try:
            if len(Y)>len(X):
                Y_ =  Y[X.index]
                if Y_.isnull().any():
                    return np.nan
                res = regress.fit(X.values.reshape(-1, 1), Y_.values.reshape(-1, 1)).coef_[0]
            else:
                # if Y.isnull().any():
                #     return np.nan
                res = regress.fit(X.values.reshape(-1, 1), Y.values.reshape(-1, 1)).coef_[0]
        except Exception as e:
            print(e)
            return np.nan
        return res
    return xs.rolling(window=n, min_periods=n).apply(lambda x:reg(x,y))


def COVIANCE(A,B,d):
    se = pd.Series(np.arange(len(A.index)),index=A.index)
    se = se.rolling(5).apply(lambda x: A.iloc[x].cov(B.iloc[x]))
    return se

def CORR(A,B,d):
    se = pd.Series(np.arange(len(A.index)),index=A.index)
    se = se.rolling(5).apply(lambda x: A.iloc[x].corr(B.iloc[x]))
    return se

def excute_for_multidates(data, func, level=0, **pramas):
    return data.groupby(level=level, group_keys=False).apply(func,**pramas)


def alpha001(data, dependencies=['close','Open','volume'], max_window=6):
    # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
    rank_sizenl = np.log(data['volume']).diff(1).rank(axis=0, pct=True)
    rank_ret = (data['close'] / data['Open']) .rank(axis=0, pct=True)
    rel = rank_sizenl.rolling(window=6,min_periods=6).corr(rank_ret) * (-1)
    return rel
    
def alpha002(data, dependencies=['close','low','high'], max_window=2):
    # -1*delta(((close-low)-(high-close))/(high-low),1)
    win_ratio = (2*data['close']-data['low']-data['high'])/(data['high']-data['low'])
    return win_ratio.diff(1) * (-1)

def alpha003(data, dependencies=['close','low','high'], max_window=6):
    # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
    # \u8fd9\u91ccSUM\u5e94\u8be5\u4e3aTSSUM
    condition2 = data['close'].diff(periods=1) > 0.0
    condition3 = data['close'].diff(periods=1) < 0.0
    alpha1 = data['close'][condition2] - np.minimum(data['close'][condition2].shift(1), data['low'][condition2])
    alpha2 = data['close'][condition3] - np.maximum(data['close'][condition3].shift(1), data['high'][condition3])
    alpha = pd.concat([alpha1,alpha2]).sort_index()
    return alpha.rolling(window=6,min_periods=6).sum() * (-1)

def alpha004(data, dependencies=['close','volume'], max_window=20):
    # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))
    # ?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))
    #     ?1:(1<=(VOLUME/MEAN(VOLUME,20))
    #       ?1:-1))
#STD(CLOSE,8)：过去8天的收盘价的标准差；VOLUME：成交量；MEAN(VOLUME,20);过去20天的均值        
    alpha = pd.Series(-1,data.index,dtype=np.dtype('int8'))
    close_mean_8 = MEAN(data['close'],8)
    close_mean_2 = MEAN(data['close'],2)
    close_std_8 = STD(data['close'],8)
    volume_mean_20 = MEAN(data['volume'],20)
    
    # alpha[(close_mean_8 + close_std_8) < close_mean_2] = -1 #这句没意义，因为默认已经是-1了
    alpha[close_mean_2 < (close_mean_8-close_std_8)] = 1
    alpha[1 <= (data['volume']/volume_mean_20)] = 1
    
    return  alpha

def alpha005(data, dependencies=['volume', 'high'], max_window=3):
    # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
    ts_volume = data['volume'].rolling(window=5,min_periods=5).apply(lambda x: stats.rankdata(x)[-1]/5.0)
    ts_high = data['high'].rolling(window=5,min_periods=5).apply(lambda x: stats.rankdata(x)[-1]/5.0)
    corr_ts = ts_volume.rolling(window=5, min_periods=5).corr(ts_high)
    # alpha = corr_ts.iloc[-3:].max(axis=0) * (-1)
    alpha = corr_ts.rolling(window=3, min_periods=3).max() * (-1)
    return alpha

def alpha006(data, dependencies=['Open', 'high'], max_window=4):
    # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
    data_mix = data['Open']*0.85+data['high']*0.15
    alpha=pd.Series(np.nan,index=data_mix.index)
    alpha[data_mix.diff(4)>1] = 1
    alpha[data_mix.diff(4)==1] = 0
    alpha[data_mix.diff(4)<1] = -1
    alpha=alpha.rolling(window=4, min_periods=4).apply(lambda x:x.rank(pct=True)[-1])
    return alpha*-1


def alpha007(data, dependencies=['volume', 'amount', 'close'], max_window=3):
    # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
    # 感觉MAX应该为TSMAX
    vwap = data['vwap']
    part1 = (vwap - data['close']).rolling(window=3,min_periods=3).max().rank(axis=0, pct=True)
    part2 = (vwap - data['close']).rolling(window=3,min_periods=3).min().rank(axis=0, pct=True)
    part3 = data['volume'].diff(3).rank(axis=0, pct=True)
    alpha = (part1 + part2) * part3
    return alpha

def alpha008(data, dependencies=['volume', 'amount', 'high', 'low'], max_window=4):
    # -1*RANK(DELTA((HIGH+LOW)/10+VWAP*0.8,4))
    # 受股价单价影响,反转
    vwap = data['vwap']
    ma_price = data['high']*0.1 + data['low']*0.1 + vwap*0.8
    alpha = ma_price.diff(max_window) * -1
    return alpha

def alpha009(data, dependencies=['high', 'low', 'volume'], max_window=8):
    # SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)
    part1 = (data['high']+data['low'])*0.5-(data['high'].shift(1)+data['low'].shift(1))*0.5
    part2 = part1 * (data['high']-data['low']) / data['volume']
    alpha = part2.ewm(adjust=False, alpha=float(2)/7, ignore_na=False).mean()
    return alpha

def alpha010(data, dependencies=['close'], max_window=25):
    # RANK(MAX(((RET<0)?STD(RET,20):CLOSE)^2,5))
    # 没法解释,感觉MAX应该为TSMAX
    ret = data['close'].pct_change(periods=1)
    part1 = ret.rolling(window=20, min_periods=20).std()
    condition = (ret >= 0.0)
    part1[condition] = data['close'][condition]
    alpha = (part1 ** 2).rolling(window=5,min_periods=5).max().rank(axis=0, pct=True)
    return alpha
    
def alpha011(data, dependencies=['close','low','high','volume'], max_window=6):
    # SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)
    # 近6天获利盘比例
    alpha = ((data['close']-data['low'])-(data['high']-data['close']))/(data['high']-data['low'])
    alpha = alpha*data['volume']
    return alpha.rolling(window=max_window, min_periods=max_window).sum()

def alpha012(data, dependencies=['Open','close','volume', 'amount'], max_window=10):
    # RANK(OPEN-MA(VWAP,10))*RANK(ABS(CLOSE-VWAP))*(-1)
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    part1 = (data['Open']-vwap.rolling(window=max_window).mean()).rank(pct=True)
    part2 = abs(data['close']-vwap).rank(axis=0, pct=True)
    alpha = part1 * part2 * (-1)
    return alpha

def alpha013(data, dependencies=['high','low','volume', 'amount'], max_window=1):
    # ((HIGH*LOW)^0.5)-VWAP
    # 要注意VWAP/price是否复权
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    alpha = np.sqrt(data['high'] * data['low']) - vwap
    return alpha

def alpha014(data, dependencies=['close'], max_window=5):
    # CLOSE-DELAY(CLOSE,5)
    # 与股价相关，利好茅台
    return data['close'].diff(max_window)

def alpha015(data, dependencies=['Open', 'close'], max_window=2):
    # OPEN/DELAY(CLOSE,1)-1
    # 跳空高开/低开
    return (data['Open']/data['close'].shift(1)-1.0)

def alpha016(data, dependencies=['volume', 'amount'], max_window=5):
    # (-1*TSMAX(RANK(CORR(RANK(VOLUME),RANK(VWAP),5)),5))
    # 感觉其中有个TSRANK
    vwap = data['vwap']
    corr_vol_vwap = data['volume'].rank(axis=0, pct=True).rolling(window=5,min_periods=5).corr(vwap.rank(axis=0, pct=True))
    alpha = corr_vol_vwap.rolling(window=5,min_periods=5).apply(lambda x: stats.rankdata(x)[-1]/5.0)
    alpha = alpha.rolling(window=5,min_periods=5).max() 
    return alpha * (-1)


def alpha017(data, dependencies=['close', 'volume', 'amount'], max_window=16):
    # RANK(VWAP-MAX(VWAP,15))^DELTA(CLOSE,5)
    vwap = data['vwap']
    delta_price = data['close'].diff(5)
    alpha = (vwap-vwap.rolling(window=15,min_periods=15).max()).rank(axis=0, pct=True) ** delta_price
    ## 长期数据时，结果有可能会非常大，加个对数抑制一下
    return np.log(alpha)

def alpha018(data, dependencies=['close'], max_window=6):
    # CLOSE/DELAY(CLOSE,5)
    # 近5日涨幅, REVS5
    return data['close'] / data['close'].shift(5)

def alpha019(data, dependencies=['close'], max_window=6):
    # (CLOSE<DELAY(CLOSE,5)?(CLOSE/DELAY(CLOSE,5)-1):(CLOSE=DELAY(CLOSE,5)?0:(1-DELAY(CLOSE,5)/CLOSE)))
    # 类似于近五日涨幅
    condition1 = data['close'] <= data['close'].shift(5)
    alpha = pd.Series(np.nan, index= data['close'].index)
    alpha[condition1] = data['close'].pct_change(periods=5)[condition1]
    alpha[~condition1] = -data['close'].pct_change(periods=5)[~condition1]
    return alpha

def alpha020(data, dependencies=['close'], max_window=7):
    # (CLOSE/DELAY(CLOSE,6)-1)*100
    # 近6日涨幅
    return (data['close'].pct_change(periods=6) * 100.0)

def alpha021(data, max_window=10):
    # REGBETA(MEAN(CLOSE,6),SEQUENCE(6))
    a = MEAN(data['close'], 6)
    a = REGBETA(a,SEQUENCE(6),6)
    if a is np.nan:
        return pd.Series(np.nan,index=data.index)
    return a

def alpha022(data, dependencies=['close'], max_window=21):
    # SMEAN((CLOSE/MEAN(CLOSE,6)-1-DELAY(CLOSE/MEAN(CLOSE,6)-1,3)),12,1)
    # 猜SMEAN是SMA
    ratio = data['close'] / data['close'].rolling(window=6,min_periods=6).mean() - 1.0
    alpha = ratio.diff(3).ewm(adjust=False, alpha=float(1)/12, min_periods=12, ignore_na=False).mean()
    return alpha
    
def alpha023(data, dependencies=['close'], max_window=40):
    # SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) /
    # (SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))
    # *100
    prc_std = data['close'].rolling(window=20, min_periods=20).std()
    condition1 = data['close'] > data['close'].shift(1)
    part1 = prc_std.copy(deep=True)
    part2 = prc_std.copy(deep=True)
    part1[~condition1] = 0.0
    part2[condition1] = 0.0
    alpha = part1.ewm(adjust=False, alpha=float(1)/20, min_periods=20, ignore_na=False).mean() / (part1.ewm(adjust=False, alpha=float(1)/20, min_periods=20, ignore_na=False).mean() + part2.ewm(adjust=False, alpha=float(1)/20, min_periods=20, ignore_na=False).mean()) * 100
    return alpha

def alpha024(data, dependencies=['close'], max_window=10):
    # SMA(CLOSE-DELAY(CLOSE,5),5,1)
    return data['close'].diff(5).ewm(adjust=False, alpha=float(1)/5, min_periods=5, ignore_na=False).mean()

def alpha025(data, dependencies=['close', 'volume'], max_window=251):
    # (-1*RANK(DELTA(CLOSE,7)*(1-RANK(DECAYLINEAR(VOLUME/MEAN(VOLUME,20),9)))))*(1+RANK(SUM(RET,250)))
    w = np.array(range(1, 10))
    w = w/w.sum()
    ret = data['close'].pct_change(periods=1)
    part1 = data['close'].diff(7)
    part2 = data['volume']/(data['volume'].rolling(window=20,min_periods=20).mean())
    part2 = 1.0 - part2.rolling(window=9, min_periods=9).apply(lambda x: np.dot(x, w)).rank(axis=0, pct=True)
    part3 = 1.0 + ret.rolling(window=250, min_periods=250).sum().rank(axis=0, pct=True)
    alpha = (-1.0) * (part1 * part2).rank(axis=0, pct=True) * part3
    return alpha

def alpha026(data, dependencies=['close', 'amount', 'volume'], max_window=235):
    # (SUM(CLOSE,7)/7-CLOSE+CORR(VWAP,DELAY(CLOSE,5),230))
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    part1 = data['close'].rolling(window=7, min_periods=7).mean() - data['close']
    part2 = vwap.rolling(window=230, min_periods=230).corr(data['close'].shift(5))
    return (part1 + part2)

def alpha027(data, dependencies=['close'], max_window=18):
    # WMA((CLOSE-DELTA(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
    part1 = data['close'].pct_change(periods=3) * 100.0 + data['close'].pct_change(periods=6) * 100.0
    # w = preprocessing.normalize(np.array([i for i in range(1, 13)]),norm='l1',axis=1).reshape(-1)
    w=np.array(range(1,13))
    w = w/w.sum()
    alpha = part1.rolling(window=12, min_periods=12).apply(lambda x: np.dot(x, w))
    return alpha

def alpha028(data, dependencies=['KDJ_J'], max_window=13):
    # 3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
    # -2*SMA(SMA((CLOSE-TSMIN(LOW,9))/( TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
    # 就是KDJ_J
    temp1=data['close']-data['low'].rolling(9).min()
    temp2=data['high'].rolling(9).max()-data['low'].rolling(9).min()
    temp3=SMA(temp1*100/temp2,3,1)
    part1=3*temp3
    part2=2*pd.DataFrame.ewm(temp3,alpha=1.0/3).mean()
    alpha=part1-part2
    return alpha

def alpha029(data, dependencies=['close', 'volume'], max_window=7):
    # (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
    # 获利成交量
    return (data['close'].pct_change(periods=6)*data['volume'])

def alpha030(data, dependencies=['close', 'pb', 'cap'], max_window=81):
    # WMA((REGRESI(RET,MKT,SMB,HML,60))^2,20)
    # 即特质性收益
    # MKT 为市值加权的市场平均收益率，
    # SMB 为市值最小的30%的股票的平均收益减去市值最大的30%的股票的平均收益，
    # HML 为pb最高的30%的股票的平均收益减去pb最低的30%的股票的平均收益    ret = data['close'].pct_change(periods=1).fillna(0.0)
    if not 'code' in data.index.names:
        print('alpha030 needs stocks more than one')
        return None
    elif len(data.index.get_level_values('code').unique())<=1:
        print('alpha030 needs stocks more than one')

    def make_factors(d):
        ret = d['returns']
        mkt_ret = (ret * d['cap']).sum() / d['cap'].sum()
        me30 = (d['cap'] <= d['cap'].quantile(0.3))
        me70 = (d['cap'] >= d['cap'].quantile(0.7))
        pb30 = (d['pb'] <= d['pb'].quantile(0.3))
        pb70 = (d['pb'] >= d['pb'].quantile(0.7))
        smb_ret = ret[me30].mean(skipna=True) - ret[me70].mean(skipna=True)
        hml_ret = ret[pb70].mean(skipna=True) - ret[pb30].mean(skipna=True)
        xs = pd.DataFrame(np.tile([mkt_ret, smb_ret, hml_ret], (len(d.index),1)), index=d.index, columns=['mkt_ret', 'smb_ret', 'hml_ret'])
        return pd.concat([xs,ret],axis=1)

    factors = excute_for_multidates(data, lambda x: make_factors(x),level=0)

    fa_ = factors.dropna()
    date_index = fa_.index.get_level_values(0).unique()
    regressor = linear_model.LinearRegression(fit_intercept=True)
    def reg(x, y):
        try:
            resid = y-regressor.fit(x,y).predict(x)
        except Exception as e:
            print(e)
            return [np.nan]*len(y)
        return resid.reshape(-1)
    
    results_tmp = []
    window=60
    for day in range(window,len(date_index)+1):
        date_range = date_index[day-window:day]
        factor_slice = fa_.loc[date_range]
        xs = factor_slice[['mkt_ret','smb_ret','hml_ret']].values
        y = factor_slice['returns'].values.reshape(-1,1)
        res = pd.Series(reg(xs,y),index=factor_slice.index)
        res = res.loc[[factor_slice.index.get_level_values(0)[-1]]]
        results_tmp.append(res)

    residual = pd.Series(np.nan,index=factors.index)
    temp = pd.concat(results_tmp,axis=0)
    residual[temp.index]=temp
    
    # w = preprocessing.normalize(np.array([i for i in range(1, 21)]).reshape(-1, 1), norm='l1', axis=0).reshape(-1)
    w = np.array(range(1, 21))
    w = w/w.sum()
    
    residual = residual ** 2
    final = excute_for_multidates(residual, lambda x: x.rolling(window=20, min_periods=20).apply(lambda x: np.dot(x, w)),level=1)

    return final

def alpha031(data, dependencies=['close'], max_window=12):
    # (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
    return ((data['close']/data['close'].rolling(window=12,min_periods=12).mean()-1.0)*100)

def alpha032(data, dependencies=['high', 'volume'], max_window=6):
    # (-1*SUM(RANK(CORR(RANK(HIGH),RANK(VOLUME),3)),3))
    # 量价齐升/反转
    part1 = data['high'].rank(axis=0, pct=True).rolling(window=3, min_periods=3).corr(data['volume'].rank(axis=0, pct=True))
    alpha = part1.rank(axis=0, pct=True).rolling(window=3, min_periods=3).sum() * (-1)
    return alpha

def alpha033(data, dependencies=['low', 'close', 'volume'], max_window=241):
    # (-1*TSMIN(LOW,5)+DELAY(TSMIN(LOW,5),5))*RANK((SUM(RET,240)-SUM(RET,20))/220)*TSRANK(VOLUME,5)
    part1 = data['low'].rolling(window=5, min_periods=5).min().diff(5) * (-1)
    ret = data['close'].pct_change(periods=1)
    part2 = ((ret.rolling(window=240, min_periods=240).sum() - ret.rolling(window=20, min_periods=20).sum()) / 220).rank(axis=0, pct=True)
    part3 = data['volume'].iloc[-5:].rank(axis=0, pct=True)
    alpha = part1 * part2 * part3
    return alpha

def alpha034(data, dependencies=['close'], max_window=12):
    # MEAN(CLOSE,12)/CLOSE
    return (data['close'].rolling(window=12, min_periods=12).mean() / data['close'])

def alpha035(data, dependencies=['Open', 'close', 'volume'], max_window=24):
    # (MIN(RANK(DECAYLINEAR(DELTA(OPEN,1),15)),RANK(DECAYLINEAR(CORR(VOLUME,OPEN*0.65+CLOSE*0.35,17),7)))*-1)
    # 猜后一项OPEN为CLOSE
    w7 =np.array(range(1, 8))
    w7 = w7/w7.sum()
    w15 = np.array(range(1, 16))
    w15 = w15/w15.sum()
    part1 = data['Open'].diff(periods=1).rolling(window=15, min_periods=15).apply(lambda x: np.dot(x, w15)).rank(axis=0, pct=True)
    part2 = (data['Open']*0.65+data['close']*0.35).rolling(window=17, min_periods=17).corr(data['volume']).rolling(window=7, min_periods=7).apply(lambda x: np.dot(x, w7)).rank(axis=0, pct=True)
    alpha = np.minimum(part1, part2) * (-1)
    return alpha

def alpha036(data, dependencies=['amount', 'volume'], max_window=9):
    # RANK(SUM(CORR(RANK(VOLUME),RANK(VWAP),6),2))
    # 量价齐升, TSSUM
    vwap = data['vwap']
    part1 = data['volume'].rank(axis=0, pct=True).rolling(window=6,min_periods=6).corr(vwap.rank(axis=0, pct=True))
    alpha = part1.rolling(window=2, min_periods=2).sum().rank(axis=0, pct=True)
    return alpha

def alpha037(data, dependencies=['Open', 'close'], max_window=16):
    # (-1*RANK(SUM(OPEN,5)*SUM(RET,5)-DELAY(SUM(OPEN,5)*SUM(RET,5),10)))
    part1 = data['Open'].rolling(window=5, min_periods=5).sum() * (data['close'].pct_change(periods=1).rolling(window=5, min_periods=5).sum())
    alpha = part1.diff(periods=10) * (-1)
    return alpha
    
def alpha038(data, dependencies=['high'], max_window=20):
    # ((SUM(HIGH,20)/20)<HIGH)?(-1*DELTA(HIGH,2)):0
    # 与股价相关，利好茅台
    condition = data['high'].rolling(window=20, min_periods=20).mean() < data['high']
    alpha = data['high'].diff(periods=2) * (-1)
    alpha[~condition] = 0.0
    return alpha


def alpha039(data, dependencies=['close', 'Open', 'amount', 'volume'], max_window=243):
    # (RANK(DECAYLINEAR(DELTA(CLOSE,2),8))-RANK(DECAYLINEAR(CORR(VWAP*0.3+OPEN*0.7,SUM(MEAN(VOLUME,180),37),14),12)))*-1
    vwap = data['vwap']
    w8 =np.array(range(1, 9))
    w8 = w8/w8.sum()
    w12 = np.array(range(1, 13))
    w12 = w12/w12.sum()
    parta = vwap * 0.3 + data['Open'] * 0.7
    partb = data['volume'].rolling(window=180, min_periods=180).mean().rolling(window=37, min_periods=37).sum()
    part1 = data['close'].diff(periods=2).rolling(window=8, min_periods=8).apply(lambda x: np.dot(x, w8)).rank(axis=0,pct=True)
    part2 = parta.rolling(window=14, min_periods=14).corr(partb).rolling(window=12, min_periods=12).apply(lambda x: np.dot(x, w12)).rank(axis=0, pct=True)
    return (part1 - part2) * (-1)

def alpha040(data,max_window=26):
    # SUM(CLOSE>DELAY(CLOSE,1)?VOLUME:0,26)/SUM(CLOSE<=DELAY(CLOSE,1)?VOLUME:0,26)*100
    # 即VR技术指标
    delay1=data.close.shift()
    condition=(data.close>delay1)
    
    vol=pd.Series(0,index=data.index)
    vol[condition]=data.volume[condition]
    vol_sum=vol.rolling(26).sum()
    
    vol1=pd.Series(0,index=data.index)
    vol1[~condition]=data.volume[~condition]
    vol1_sum=vol1.rolling(26).sum()
    alpha=vol_sum/vol1_sum * 100
    return alpha


def alpha041(data, dependencies=['amount', 'volume'], max_window=9):
    # RANK(MAX(DELTA(VWAP,3),5))*-1
    vwap = data['vwap']
    return vwap.diff(periods=3).rolling(window=5, min_periods=5).max().rank(axis=0, pct=True) * (-1)

def alpha042(data, dependencies=['high', 'volume'], max_window=10):
    # (-1*RANK(STD(HIGH,10)))*CORR(HIGH,VOLUME,10)
    # 价稳/量价齐升
    part1 = data['high'].rolling(window=10,min_periods=10).std().rank(axis=0,pct=True) * (-1)
    part2 = data['high'].rolling(window=10,min_periods=10).corr(data['volume'])
    return (part1 * part2)

def alpha043(data, dependencies=['OBV6'], max_window=7):
    # (SUM(CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0),6))
    # 即OBV6指标
    delay1=data.close.shift()
    temp=pd.Series(0,index=data.index)
    condition1=(data.close>delay1)
    condition2=(data.close<delay1)
    temp[condition1] = data.volume[condition1]
    temp[condition2] = data.volume[condition2] * -1
    alpha=temp.rolling(6).sum()
    return alpha

def alpha044(data, dependencies=['amount', 'volume', 'low'], max_window=29):
    # (TSRANK(DECAYLINEAR(CORR(LOW,MEAN(VOLUME,10),7),6),4)+TSRANK(DECAYLINEAR(DELTA(VWAP,3),10),15))
    n=6
    m=10
    seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]   #Decaylinear 1
    seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]   #Decaylinear 2
    weight1=np.array(seq1)
    weight2=np.array(seq2)
    
    temp1=data.low.rolling(7).corr(data.volume.rolling(10).mean())
    part1=temp1.rolling(n).apply(lambda x: (x*weight1).sum())   #dataframe * numpy array
    part1=part1.rolling(4).apply(lambda x: x.rank(pct=True)[-1])
    
    temp2=data.vwap.diff(3)
    part2=temp2.rolling(m).apply(lambda x: (x*weight2).sum())
    part2=part2.rolling(5).apply(lambda x: x.rank(pct=True)[-1])
    alpha=part1 + part2 
    return alpha
    
def alpha045(data, dependencies=['Open', 'close', 'amount', 'volume'], max_window=165):
    # (RANK(DELTA(CLOSE*0.6+OPEN*0.4,1))*RANK(CORR(VWAP,MEAN(VOLUME,150),15)))
    vwap = data['vwap']
    part1 = (data['close'] * 0.6 + data['Open'] * 0.4).diff(periods=1).rank(axis=0,pct=True)
    part2 = (vwap.rolling(window=15,min_periods=15).corr(data['volume'].rolling(window=150,min_periods=150).mean())).rank(axis=0,pct=True)
    return part1*part2

def alpha046(data, dependencies=['BBIC'], max_window=24):
    # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
    # 即BBIC技术指标
    part1=[3,6,12,24]
    part2=[data['close'].rolling(window=x,min_periods=x).mean() for x in part1]
    return sum(part2)/data['close']*4

def alpha047(data, dependencies=['close', 'low', 'high'], max_window=15):
    # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)
    # RSV技术指标变种
    part1 = (data['high'].rolling(window=6,min_periods=6).max()-data['close']) /  (data['high'].rolling(window=6,min_periods=6).max()- data['low'].rolling(window=6,min_periods=6).min()) * 100
    alpha = part1.ewm(adjust=False, alpha=float(1)/9, min_periods=0, ignore_na=False).mean()
    return alpha

def alpha048(data, dependencies=['close', 'volume'], max_window=20):
    # -1*RANK(SIGN(CLOSE-DELAY(CLOSE,1))+SIGN(DELAY(CLOSE,1)-DELAY(CLOSE,2))+SIGN(DELAY(CLOSE,2)-DELAY(CLOSE,3)))*SUM(VOLUME,5)/SUM(VOLUME,20)
    # 下跌缩量
    diff1 = data['close'].diff(1)
    part1 = (np.sign(diff1) + np.sign(diff1.shift(1)) + np.sign(diff1.shift(2))).rank(axis=0, pct=True)
    part2 = data['volume'].rolling(window=5, min_periods=5).sum() / data['volume'].rolling(window=20, min_periods=20).sum()
    return (part1 * part2) * (-1)

def alpha049(data, dependencies=['high', 'low'], max_window=13):
    # SUM(HIGH+LOW>=DELAY(HIGH,1)+DELAY(LOW,1)?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1))),12)/
    # (SUM(HIGH+LOW>=DELAY(HIGH,1)+DELAY(LOW,1)?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1))),12)+
    # SUM(HIGH+LOW<=DELAY(HIGH,1)+DELAY(LOW,1)?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1))),12))
    condition1 = (data['high'] + data['low']) >= (data['high'] + data['low']).shift(1)
    condition2 = (data['high'] + data['low']) <= (data['high'] + data['low']).shift(1)
    part1 = pd.Series(0, index=data.index)
    part2 = pd.Series(0, index=data.index)
    part1[~condition1] = np.maximum(abs(data['high'].diff(1)[~condition1]), abs(data['low'].diff(1)[~condition1]))
    part2[~condition2] = np.maximum(abs(data['high'].diff(1)[~condition2]), abs(data['low'].diff(1)[~condition2]))
    alpha = part1.rolling(window=12,min_periods=12).sum() / (part1.rolling(window=12,min_periods=12).sum() + part2.rolling(window=12,min_periods=12).sum())
    return alpha

def alpha050(data, dependencies=['high', 'low'], max_window=13):
    # SUM(HIGH+LOW<=DELAY(HIGH,1)+DELAY(LOW,1)?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1))),12)/
    # (SUM(HIGH+LOW<=DELAY(HIGH,1)+DELAY(LOW,1)?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1))),12)
    # +SUM(HIGH+LOW>=DELAY(HIGH,1)+DELAY(LOW,1)?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1))),12))
    # -SUM(HIGH+LOW>=DELAY(HIGH,1)+DELAY(LOW,1)?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1))),12)/
    # (SUM(HIGH+LOW>=DELAY(HIGH,1)+DELAY(LOW,1)?0: MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1))),12)
    # +SUM(HIGH+LOW<=DELAY(HIGH,1)+DELAY(LOW,1)?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1))),12))
    condition1 = (data['high'] + data['low']) >= (data['high'] + data['low']).shift(1)
    condition2 = (data['high'] + data['low']) <= (data['high'] + data['low']).shift(1)
    part = np.maximum(abs(data['high'].diff(1)), abs(data['low'].diff(1)))
    a=(part*condition2).rolling(12).sum()
    b=(part*condition1).rolling(12).sum()
    ab=a+b
    return a/ab-b/ab

def alpha051(data, dependencies=['high', 'low'], max_window=13):
    # SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/
    # (SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
    # +SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    condition1 = (data['high'] + data['low']) <= (data['high'] + data['low']).shift(1)
    condition2 = (data['high'] + data['low']) >= (data['high'] + data['low']).shift(1)
    part1 = pd.Series(0, index=data.index)
    part2 = pd.Series(0, index=data.index) 
    part1[~condition1] = np.maximum(abs(data['high'].diff(1)[~condition1]), abs(data['low'].diff(1)[~condition1]))
    part2[~condition2] = np.maximum(abs(data['high'].diff(1)[~condition2]), abs(data['low'].diff(1)[~condition2]))
    alpha = part1.rolling(window=12,min_periods=12).sum() / (part1.rolling(window=12,min_periods=12).sum() + part2.rolling(window=12,min_periods=12).sum())
    return alpha

def alpha052(data, dependencies=['high', 'low', 'close'], max_window=27):
    # SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-L),26)*100
    ma = (data['high'] + data['low'] + data['close']) / 3.0
    part1 = (np.maximum(0.0, (data['high'] - ma.shift(1)))).rolling(window=26, min_periods=26).sum()
    part2 = (np.maximum(0.0, (ma.shift(1) - data['low']))).rolling(window=26, min_periods=26).sum()
    return part1 / part2 * 100.0

def alpha053(data, dependencies=['close'], max_window=13):
    # COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
    return ((data['close'].diff(1) > 0.0).rolling(window=12, min_periods=12).sum() / 12.0 * 100)

def alpha054(data, dependencies=['close', 'Open'], max_window=10):
    # (-1*RANK(STD(ABS(CLOSE-OPEN))+CLOSE-OPEN+CORR(CLOSE,OPEN,10)))
    # 注，这里STD没有指明周期
    part1 = abs(data['close']-data['Open']).rolling(window=10, min_periods=10).std() + data['close'] - data['Open'] + data['close'].rolling(window=10, min_periods=10).corr(data['Open'])
    return part1.rank(axis=0, pct=True) * (-1)

def alpha055(data, dependencies=['Open', 'low', 'close', 'high'], max_window=21):
    # SUM(16*(CLOSE+(CLOSE-OPEN)/2-DELAY(OPEN,1))/
    # ((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) ? 
    # ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:
    # (ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1)) ?
    # ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:
    # ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))
    # *MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)
    part1 = data['close'] * 1.5 - data['Open'] * 0.5 - data['Open'].shift(1)
    part2 = abs(data['high']-data['close'].shift(1)) + abs(data['low']-data['close'].shift(1)) / 2.0 + abs(data['close']-data['Open']).shift(1) / 4.0
    condition1 = np.logical_and(abs(data['high']-data['close'].shift(1)) > abs(data['low']-data['close'].shift(1)), 
                               abs(data['high']-data['close'].shift(1)) > abs(data['high']-data['low'].shift(1)))
    condition2 = np.logical_and(abs(data['low']-data['close'].shift(1)) > abs(data['high']-data['low'].shift(1)), 
                               abs(data['low']-data['close'].shift(1)) > abs(data['high']-data['close'].shift(1)))
    part2[~condition1 & condition2] = abs(data['low']-data['close'].shift(1)) + abs(data['high']-data['close'].shift(1)) / 2.0 + abs(data['close']-data['Open']).shift(1) / 4.0
    part2[~condition1 & ~condition2] = abs(data['high']-data['low'].shift(1)) + abs(data['close']-data['Open']).shift(1) / 4.0
    part3 = np.maximum(abs(data['high']-data['close'].shift(1)), abs(data['low']-data['close'].shift(1)))
    alpha = (part1 / part2 * part3 * 16.0).rolling(window=20, min_periods=20).sum()
    return alpha

def alpha056(data, dependencies=['Open', 'high', 'low', 'volume'], max_window=73):
    # RANK(OPEN-TSMIN(OPEN,12))<RANK(RANK(CORR(SUM((HIGH +LOW)/2,19),SUM(MEAN(VOLUME,40),19),13))^5)
    # 这里就会有随机性,0/1
    part1 = (data['Open'] - data['Open'].rolling(window=12, min_periods=12).min()).rank(axis=0, pct=True)
    t1 = (data['high']*0.5+data['low']*0.5).rolling(window=19, min_periods=19).sum()
    t2 = data['volume'].rolling(window=40,min_periods=40).mean().rolling(window=19, min_periods=19).sum()
    part2 = ((t1.rolling(window=13, min_periods=13).corr(t2).rank(axis=0, pct=True)) ** 5).rank(axis=0, pct=True)
    return part2-part1

def alpha057(data, dependencies=['KDJ_K'], max_window=11):
    # SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
    # KDJ_K
    part1 =data['close'] - data['close'].rolling(window=9, min_periods=9).min()
    part2=data['high'].rolling(window=9, min_periods=9).max()-data['low'].rolling(window=9, min_periods=9).min()
    return SMA(part1/part2*100,3,1)

def alpha058(data, dependencies=['close'], max_window=20):
    # COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
    # alpha = ((data['close'].diff(1) > 0.0).rolling(window=20, min_periods=20).sum() / 20 * 100)
    alpha = ((data['close'].diff(1) > 0.0).rolling(window=20, min_periods=20).sum() * 5)
    return alpha

def alpha059(data, dependencies=['close', 'low', 'high'], max_window=21):
    # SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)
    # 受价格尺度影响
    alpha = pd.Series(0, index=data.index) 
    condition1 = data['close'].diff(1) > 0.0
    condition2 = data['close'].diff(1) < 0.0
    alpha[condition1] = data['close'][condition1] - np.minimum(data['low'][condition1], data['close'].shift(1)[condition1])
    alpha[condition2] = data['close'][condition2] - np.maximum(data['high'][condition2], data['close'].shift(1)[condition2])
    alpha = alpha.rolling(window=20, min_periods=20).sum()
    return alpha

def alpha060(data, dependencies=['close', 'Open', 'low', 'high', 'volume'], max_window=21):
    # SUM((2*CLOSE-LOW-HIGH)./(HIGH-LOW).*VOLUME,20)
    part1 = (2*data['close']-data['low']-data['high']) / (data['high']-data['low']) * data['volume']
    return part1.rolling(window=20, min_periods=20).sum()

def alpha061(data, dependencies=['low', 'amount', 'volume'], max_window=106):
    # MAX(RANK(DECAYLINEAR(DELTA(VWAP,1),12)),RANK(DECAYLINEAR(RANK(CORR(LOW,MEAN(VOLUME,80),8)),17)))*-1
    vwap = data['vwap']
    w12 = np.array(range(1, 13))
    w12 = w12/w12.sum()
    w17 = np.array(range(1, 18))
    w17 = w17/w17.sum()
    turnover_ma = data['volume'].rolling(window=80, min_periods=80).mean()
    part1 = vwap.diff(periods=1).rolling(window=12, min_periods=12).apply(lambda x: np.dot(x, w12)).rank(axis=0, pct=True)
    part2 = (turnover_ma.rolling(window=8, min_periods=8).corr(data['low']).rank(axis=0,pct=True)).rolling(window=17, min_periods=17).apply(lambda x: np.dot(x, w17)).rank(axis=0, pct=True)
    alpha = np.maximum(part1, part2) * (-1)
    return alpha

def alpha062(data, dependencies=['volume', 'high'], max_window=5):
    # -1*CORR(HIGH,RANK(VOLUME),5)
    return data['volume'].rank(axis=0, pct=True).rolling(window=5, min_periods=5).corr(data['high']) * (-1)


def alpha063(data, dependencies=['close'], max_window=7):
    # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
    part1 = (np.maximum(data['close'].diff(1), 0.0)).ewm(adjust=False, alpha=float(1)/6, min_periods=0, ignore_na=False).mean()
    part2 = abs(data['close'].diff(1)).ewm(adjust=False, alpha=float(1)/6, min_periods=0, ignore_na=False).mean()
    return part1/part2*100.0
    
def alpha064(data, dependencies=['close', 'amount', 'volume'], max_window=93):
    # (MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),4),4)),RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE),RANK(MEAN(VOLUME,60)),4),13),14)))*-1)
    # 看上去是TSMAX
    vwap = data['vwap']
    w4 = np.array(range(1, 5))
    w4 = w4/w4.sum()
    w14 = np.array(range(1, 15))
    w14 = w14/w14.sum()
    part1 = (vwap.rank(axis=0, pct=True).rolling(window=4, min_periods=4).corr(data['volume'].rank(axis=0, pct=True))).rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, w4)).rank(axis=0, pct=True)
    part2 = (data['volume'].rolling(window=60, min_periods=60).mean().rank(axis=0, pct=True)).rolling(window=4, min_periods=4).corr(data['close'].rank(axis=0, pct=True))
    part2 = (part2.rolling(window=13, min_periods=13).max()).rolling(window=14, min_periods=14).apply(lambda x: np.dot(x, w14)).rank(axis=0,pct=True)
    alpha = np.maximum(part1, part2) * (-1)
    return alpha

def alpha065(data, dependencies=['close'], max_window=6):
    # MEAN(CLOSE,6)/CLOSE
    return (data['close'].rolling(window=6, min_periods=6).mean() / data['close'])

def alpha066(data, dependencies=['BIAS5'], max_window=6):
    # (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
    # BIAS6，用BIAS5简单替换下
    # part1=data['close'].iloc[-1]-data['close'].mean()
    close_mean = data['close'].rolling(6).mean()
    alpha = (data['close'] - close_mean)/close_mean *100
    return alpha

def alpha067(data, dependencies=['close'], max_window=25):
    # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
    # RSI24
    part1 = (np.maximum(data['close'].diff(1), 0.0)).ewm(adjust=False, alpha=float(1)/24, min_periods=0, ignore_na=False).mean()
    part2 = (abs(data['close'].diff(1))).ewm(adjust=False, alpha=float(1)/24, min_periods=0, ignore_na=False).mean()
    return (part1 / part2 * 100)

def alpha068(data, dependencies=['high', 'low', 'volume'], max_window=16):
    # SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)
    part1 = (data['high'].diff(1) * 0.5 + data['low'].diff(1) * 0.5) * (data['high'] - data['low']) / data['volume']
    return part1.ewm(adjust=False, alpha=float(2)/15, min_periods=0, ignore_na=False).mean()

def alpha069(data, dependencies=['Open', 'high', 'low'], max_window=21):
    # (SUM(DTM,20)>SUM(DBM,20)?
        #(SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20):
        #(SUM(DTM,20)=SUM(DBM,20)?0:
            #(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))
    # DTM: (OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
    # DBM: (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
    dtm=(data['Open'].diff(1) <= 0) * np.maximum(data['high']-data['Open'],data['Open'].diff(1))
    dbm=(data['Open'].diff(1) >= 0) * np.maximum(data['Open']-data['low'],data['Open'].diff(1))
    dtm_sum = dtm.rolling(window=20, min_periods=20).sum()
    dbm_sum = dbm.rolling(window=20, min_periods=20).sum()
    # if dtm_sum>dbm_sum:
    #     return (dtm_sum-dbm_sum)/dtm_sum
    # elif dtm_sum==dbm_sum:
    #     return 0
    # else:
    #     return (dtm_sum-dbm_sum)/dbm_sum
    alpha = pd.Series(np.nan, index = data.index,dtype=np.dtype('float32'))
    condition = dtm_sum>dbm_sum
    dif = dtm_sum-dbm_sum
    alpha[condition] = (dif/dtm_sum)[condition]
    alpha[~condition] = (dif/dbm_sum)[~condition]
    alpha[(dtm_sum==dbm_sum)] = 0
    return alpha

def alpha070(data, dependencies=['amount'], max_window=6):
    # STD(AMOUNT,6)
    return data['amount'].rolling(window=6, min_periods=6).std()

def alpha071(data, dependencies=['close'], max_window=25):
    # (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
    # BIAS24
    close_ma = data['close'].rolling(window=24, min_periods=24).mean()
    return (data['close'] - close_ma) / close_ma * 100

def alpha072(data, dependencies=['high', 'low', 'close'], max_window=22):
    # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
    part1 = (data['high'].rolling(window=6, min_periods=6).max() - data['close']) / (data['high'].rolling(window=6, min_periods=6).max() - data['low'].rolling(window=6,min_periods=6).min()) * 100.0
    return part1.ewm(adjust=False, alpha=float(1)/15, min_periods=0, ignore_na=False).mean()

def alpha073(data, dependencies=['amount', 'volume', 'close'], max_window=38):
    # ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR(CLOSE,VOLUME,10),16),4),5)-RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,30),4),3)))*-1)
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    w16 =np.array(range(1, 17))
    w16 = w16/w16.sum()
    w4 =np.array(range(1, 5))
    w4 = w4/w4.sum()
    w3 =np.array(range(1, 4))
    w3 = w3/w3.sum()
    part1 = (data['close'].rolling(window=10, min_periods=10).corr(data['volume'])).rolling(window=16, min_periods=16).apply(lambda x: np.dot(x, w16))
    part1 = (part1.rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, w4))).rolling(window=5, min_periods=5).apply(lambda x: stats.rankdata(x)[-1]/5.0)
    part2 = data['volume'].rolling(window=30, min_periods=30).mean().rolling(window=4, min_periods=4).corr(vwap)
    part2 = part2.rolling(window=3, min_periods=3).apply(lambda x: np.dot(x, w3)).rank(axis=0, pct=True)
    return (part1 - part2) * (-1)


def alpha074(data, dependencies=['low', 'amount', 'volume'], max_window=68):
    # RANK(CORR(SUM(LOW*0.35+VWAP*0.65,20),SUM(MEAN(VOLUME,40),20),7))+RANK(CORR(RANK(VWAP),RANK(VOLUME),6))
    # vwap = data['amount'] / (data['volume']*100) 
    vwap = data['vwap']
    part1 = ((data['low'] * 0.35 + vwap * 0.65).rolling(window=20, min_periods=20).sum()).rolling(window=7, min_periods=7).corr((data['volume'].rolling(window=40,min_periods=40).mean()).rolling(window=20, min_periods=20).sum()).rank(axis=0, pct=True)
    part2 = (vwap.rank(axis=0,pct=True).rolling(window=6, min_periods=6).corr(data['volume'].rank(axis=0, pct=True))).rank(axis=0, pct=True)
    return part1 + part2

def alpha075(data, dependencies=['close', 'Open','bm_index_open', 'bm_index_close'], max_window=51):
    # COUNT(CLOSE>OPEN & BANCHMARK_INDEX_CLOSE<BANCHMARK_INDEX_OPEN,50)/COUNT(BANCHMARK_INDEX_CLOSE<BANCHMARK_INDEX_OPEN,50)
    # 简化为等权benchmark
    # bm = (data['close'].mean() < data['Open'].mean())
    close_bm = data['bm_index_close']
    open_bm = data['bm_index_open']
    bm_den = close_bm < open_bm
    alpha = np.logical_and(data['close'] > data['Open'], bm_den).rolling(window=50, min_periods=50).sum() / bm_den.rolling(window=50, min_periods=50).sum()
    alpha = alpha.fillna(0)
    alpha[0:49] = np.nan
    return alpha

def alpha076(data, dependencies=['close', 'volume'], max_window=21):
    # STD(ABS(CLOSE/DELAY(CLOSE,1)-1)/VOLUME,20)/MEAN(ABS(CLOSE/DELAY(CLOSE,1)-1)/VOLUME,20)
    ret_vol = abs(data['close'].pct_change(periods=1))/data['volume']
    return (ret_vol.rolling(window=20, min_periods=20).std() / ret_vol.rolling(window=20, min_periods=20).mean())

def alpha077(data, dependencies=['low', 'high', 'amount', 'volume'], max_window=50):
    # MIN(RANK(DECAYLINEAR(HIGH*0.5+LOW*0.5-VWAP,20)),RANK(DECAYLINEAR(CORR(HIGH*0.5+LOW*0.5,MEAN(VOLUME,40),3),6)))
    # vwap = data['amount'] / (data['volume']*100) 
    vwap = data['vwap']
    w6 = np.array(range(1, 7))
    w6 = w6/w6.sum()
    w20 = np.array(range(1, 21))
    w20 = w20/w20.sum()
    part1 = (data['high'] * 0.5 + data['low'] * 0.5 - vwap).rolling(window=20, min_periods=20).apply(lambda x: np.dot(x, w20)).rank(axis=0, pct=True)
    part2 = ((data['high'] * 0.5 + data['low'] * 0.5).rolling(window=3, min_periods=3).corr(data['volume'].rolling(window=40, min_periods=40).mean())).rolling(window=6, min_periods=6).apply(lambda x: np.dot(x, w6)).rank(axis=0, pct=True)
    return np.minimum(part1, part2)
    
def alpha078(data, dependencies=['CCI10'], max_window=12):
    # ((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))
    #/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))
    # 相当于是CCI12, 用CCI10替代
    part1=(data['high']+data['low']+data['close'])/3
    part2=part1-part1.rolling(window=12, min_periods=12).mean()
    part3=(data['close']-part1.rolling(window=12, min_periods=12).mean()).abs().mean()*0.015
    return part2/part3

def alpha079(data, dependencies=['close', 'Open'], max_window=13):
    # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
    # 就是RSI12
    part1 = (np.maximum(data['close'].diff(1), 0.0)).ewm(adjust=False, alpha=float(1)/12, min_periods=0, ignore_na=False).mean()
    part2 = (abs(data['close'].diff(1))).ewm(adjust=False, alpha=float(1)/12, min_periods=0, ignore_na=False).mean()
    return part1 / part2 * 100

def alpha080(data, dependencies=['volume'], max_window=6):
    # (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
    return (data['volume'].pct_change(periods=5) * 100.0)

def alpha081(data, dependencies=['volume'], max_window=21):
    # SMA(VOLUME,21,2)
    return data['volume'].ewm(adjust=False, alpha=float(2)/21, min_periods=0, ignore_na=False).mean()

def alpha082(data, dependencies=['low', 'high', 'close'], max_window=26):
    # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)
    # RSV技术指标变种
    part1 = (data['high'].rolling(window=6,min_periods=6).max()-data['close']) / (data['high'].rolling(window=6,min_periods=6).max()-data['low'].rolling(window=6,min_periods=6).min()) * 100
    alpha = part1.ewm(adjust=False, alpha=float(1)/20, min_periods=0, ignore_na=False).mean()
    return alpha

def alpha083(data, dependencies=['high', 'volume'], max_window=5):
    # (-1*RANK(COVIANCE(RANK(HIGH),RANK(VOLUME),5)))
    alpha = COVIANCE(data['high'].rank(pct=True),data['volume'].rank(pct=True),5)*-1
    return alpha

def alpha084(data, dependencies=['close', 'volume'], max_window=21):
    # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)
    part1 = np.sign(data['close'].diff(1)) * data['volume']
    return part1.rolling(window=20, min_periods=20).sum()

def alpha085(data, dependencies=['close', 'volume'], max_window=40):
    # TSRANK(VOLUME/MEAN(VOLUME,20),20)*TSRANK(a-1*DELTA(CLOSE,7),8)
    part1 = (data['volume'] / data['volume'].rolling(window=20,min_periods=20).mean())
    part1 = part1.rolling(window=20,min_periods=20).apply(lambda x:x.rank(pct=True)[-1])
    
    part2 = (data['close'].diff(7) * (-1)).rolling(window=8,min_periods=8).apply(lambda x:x.rank(pct=True)[-1])
    return round(part1*part2, 8)

def alpha086(data, dependencies=['close'], max_window=21):
    # ((0.25<((DELAY(CLOSE,20)-DELAY(CLOSE,10))/10-(DELAY(CLOSE,10)-CLOSE)/10))?-1:((((DELAY(CLOSE,20)-DELAY(CLOSE,10))/10-(DELAY(CLOSE,10)-CLOSE)/10)<0)?1:(DELAY(CLOSE,1)-CLOSE)))
    condition1 = (data['close'].shift(20) * 0.1 + data['close'] * 0.1 - data['close'].shift(10) * 0.2) > 0.25
    condition2 = (data['close'].shift(20) * 0.1 + data['close'] * 0.1 - data['close'].shift(10) * 0.2) < 0.0
    alpha = pd.Series(np.nan,index=data.index)
    alpha[condition1] = -1.0
    alpha[~condition1 & condition2] = 1.0
    alpha[~condition1 & ~condition2] = data['close'].diff(1)[~condition1 & ~condition2] * (-1)
    return round(alpha,8)

def alpha087(data, dependencies=['amount', 'volume', 'low', 'high', 'Open'], max_window=18):
    # (RANK(DECAYLINEAR(DELTA(VWAP,4),7))+TSRANK(DECAYLINEAR((LOW-VWAP)/(OPEN-(HIGH+LOW)/2),11),7))*-1
    # vwap = data['amount'] / (data['volume']*100) 
    vwap = data['vwap']
    w7 = np.array(range(1, 8))
    w7 = w7/w7.sum()
    w11 = np.array(range(1, 12))
    w11 = w11/w11.sum()
    part1 = (vwap.diff(4).rolling(window=7, min_periods=7).apply(lambda x: np.dot(x, w7))).rank(pct=True)
    part2 = (data['low']-vwap)/(data['Open']-data['high']*0.5-data['low']*0.5)
    part2 = part2.rolling(window=11, min_periods=11).apply(lambda x: np.dot(x, w11))
    part2 = part2.rolling(window=7, min_periods=7).apply(lambda x: x.rank(pct=True)[-1])
    return (part1 + part2) * (-1)

def alpha088(data, dependencies=['REVS20'], max_window=20):
    # (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
    # 就是REVS20
    # return (data['close'].iloc[-1]-data['close'].iloc[-20])/data['close'].iloc[-20]*100
    alpha = data['close'].rolling(20).apply(lambda x: (x[-1]-x[0])/x[0]*100)
    return alpha

def alpha089(data, dependencies=['close'], max_window=37):
    # 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
    part1 = data['close'].ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean() - data['close'].ewm(adjust=False, alpha=float(2)/27, min_periods=0, ignore_na=False).mean()
    alpha = (part1 - part1.ewm(adjust=False, alpha=float(2)/10, min_periods=0, ignore_na=False).mean()) * 2.0
    return alpha

def alpha090(data, dependencies=['amount', 'volume'], max_window=5):
    # (RANK(CORR(RANK(VWAP),RANK(VOLUME),5))*-1)
    return CORR(data['vwap'].rank(pct=True),data['volume'].rank(pct=True),5)*-1
    




def alpha091(data, dependencies=['close', 'volume', 'low'], max_window=45):
    # ((RANK(CLOSE-MAX(CLOSE,5))*RANK(CORR(MEAN(VOLUME,40),LOW,5)))*-1)
    # 感觉是TSMAX
    part1 = (data['close'] - data['close'].rolling(window=5, min_periods=5).max()).rank(axis=0, pct=True)
    part2 = (data['volume'].rolling(window=40, min_periods=40).mean()).rolling(window=5, min_periods=5).corr(data['low']).rank(pct=True)
    return (part1 * part2) * (-1)

def alpha092(data, dependencies=['close', 'amount', 'volume'], max_window=209):
    # (MAX(RANK(DECAYLINEAR(DELTA(CLOSE*0.35+VWAP*0.65,2),3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)),CLOSE,13)),5),15))*-1)
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    w3 = np.array(range(1, 4))
    w3 = w3/w3.sum()
    w5 = np.array(range(1, 6))
    w5 = w5/w5.sum()
    part1 = ((data['close'] * 0.35 + vwap * 0.65).diff(2)).rolling(window=3, min_periods=3).apply(lambda x: np.dot(x, w3)).rank(axis=0, pct=True)
    part2 = abs((data['volume'].rolling(window=180, min_periods=180).mean()).rolling(window=13, min_periods=13).corr(data['close']))
    part2 = part2.rolling(window=5, min_periods=5).apply(lambda x: np.dot(x, w5))
    part2 = part2.rolling(window=15, min_periods=15).apply(lambda x: x.rank(pct=True)[-1])
    return np.maximum(part1, part2) * (-1)

def alpha093(data, dependencies=['Open', 'low'], max_window=21):
    # SUM(OPEN>=DELAY(OPEN,1)?0:MAX(OPEN-LOW,OPEN-DELAY(OPEN,1)),20)
    condition = data['Open'].diff(1) >= 0.0
    alpha= pd.Series(0,index=data.index)
    alpha[~condition] = np.maximum(data['Open'] - data['low'], data['Open'].diff(1))[~condition]
    return alpha.rolling(window=20, min_periods=20).sum()

def alpha094(data, dependencies=['close', 'volume'], max_window=31):
    # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)
    part1 = np.sign(data['close'].diff(1)) * data['volume']
    return part1.rolling(window=30, min_periods=30).sum()

def alpha095(data, dependencies=['amount'], max_window=20):
    # STD(AMOUNT,20)
    return data['amount'].rolling(window=20, min_periods=20).std()

def alpha096(data, dependencies=['KDJ_D'], max_window=13):
    # SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
    # 就是KDJ_D
    part1 =data['close']- data['close'].rolling(window=9, min_periods=9).min()
    part2=data['high'].rolling(window=9, min_periods=9).max()-data['low'].rolling(window=9, min_periods=9).min()
    part4=SMA(part1/part2*100,3,1)
    part5=SMA(part4,3,1)
    return part5

def alpha097(data, dependencies=['VSTD10'], max_window=10):
    # STD(VOLUME,10)
    # 就是VSTD10
    return STD(data['volume'],10)

def alpha098(data, dependencies=['close'], max_window=201):
    # (DELTA(SUM(CLOSE,100)/100,100)/DELAY(CLOSE,100)<=0.05)?(-1*(CLOSE-TSMIN(CLOSE,100))):(-1*DELTA(CLOSE,3))
    condition1 = (data['close'].rolling(window=100, min_periods=100).sum() / 100).diff(periods=100) / data['close'].shift(100) <= 0.05
    alpha = (data['close'] - data['close'].rolling(window=100, min_periods=100).min()) * (-1)
    alpha[~condition1] = data['close'].diff(3)[~condition1] * (-1)
    return alpha

def alpha099(data, dependencies=['close', 'volume'], max_window=5):
    # (-1*RANK(COVIANCE(RANK(CLOSE),RANK(VOLUME),5)))
    # return COVIANCE(sorted(data['close']),sorted(data['volume']),5)*-1
    return COVIANCE(data['close'].rank(pct=True),data['volume'].rank(pct=True),5)*-1

def alpha100(data, dependencies=['VSTD20'], max_window=20):
    # STD(VOLUME,20), 就是VSTD10
    return STD(data['volume'],20)

def alpha101(data, dependencies=['amount', 'volume', 'high', 'close'], max_window=82):
    # (RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15)) < RANK(CORR(RANK(HIGH*0.1+VWAP*0.9),RANK(VOLUME),11)))*-1
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    part1 = (data['volume'].rolling(window=30, min_periods=30).mean()).rolling(window=37, min_periods=37).sum()
    part1 = (part1.rolling(window=15, min_periods=15).corr(data['close'])).rank(axis=0, pct=True)
    part2 = (data['high'] * 0.1 + vwap * 0.9).rank(axis=0, pct=True)
    part2 = (part2.rolling(window=11, min_periods=11).corr(data['volume'].rank(axis=0, pct=True))).rank(axis=0, pct=True)
    return (part2 - part1) * (-1)

def alpha102(data, dependencies=['volume'], max_window=7):
    # SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    part1 = (np.maximum(data['volume'].diff(1), 0.0)).ewm(adjust=False, alpha=float(1)/6, min_periods=0, ignore_na=False).mean()
    part2 = abs(data['volume'].diff(1)).ewm(adjust=False, alpha=float(1)/6, min_periods=0, ignore_na=False).mean()
    return (part1 / part2) * 100

def alpha103(data, dependencies=['low'], max_window=20):
    # ((20-LOWDAY(LOW,20))/20)*100
    return (20 - data['low'].rolling(window=20, min_periods=20).apply(lambda x: 19-x.argmin(axis=0))) * 5.0

def alpha104(data, dependencies=['high', 'volume', 'close'], max_window=20):
    # -1*(DELTA(CORR(HIGH,VOLUME,5),5)*RANK(STD(CLOSE,20)))
    part1 = (data['high'].rolling(window=5, min_periods=5).corr(data['volume'])).diff(5)
    part2 = (data['close'].rolling(window=20, min_periods=20).std()).rank(axis=0, pct=True)
    return (part1 * part2) * (-1)

def alpha105(data, dependencies=['Open', 'volume'], max_window=10):
    # -1*CORR(RANK(OPEN),RANK(VOLUME),10)
    alpha = (data['Open'].rank(axis=0, pct=True)).rolling(window=10, min_periods=10).corr(data['volume'].rank(pct=True))
    return alpha * (-1)

def alpha106(data, dependencies=['close'], max_window=21):
    # CLOSE-DELAY(CLOSE,20)
    return data['close'].diff(20)

def alpha107(data, dependencies=['Open', 'close', 'high', 'low'], max_window=2):
    # (-1*RANK(OPEN-DELAY(HIGH,1)))*RANK(OPEN-DELAY(CLOSE,1))*RANK(OPEN-DELAY(LOW,1))
    part1 = data['Open'] - data['high'].shift(1)
    part2 = data['Open'] - data['close'].shift(1)
    part3 = data['Open'] - data['low'].shift(1)
    return (part1 * part2 * part3) * (-1)

def alpha108(data, dependencies=['high', 'amount', 'volume'], max_window=126):
    # (RANK(HIGH-MIN(HIGH,2))^RANK(CORR(VWAP,MEAN(VOLUME,120),6)))*-1
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    part1 = (data['high'] - data['high'].rolling(window=2,min_periods=2).min()).rank(axis=0, pct=True)
    part2 = (data['volume'].rolling(window=120, min_periods=120).mean()).rolling(window=6, min_periods=6).corr(vwap).rank(axis=0, pct=True)
    return (part1 ** part2) * (-1)

def alpha109(data, dependencies=['high', 'low'], max_window=20):
    # SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)
    part1 = (data['high']-data['low']).ewm(adjust=False, alpha=float(2)/10, min_periods=0, ignore_na=False).mean()
    return (part1 / part1.ewm(adjust=False, alpha=float(2)/10, min_periods=0, ignore_na=False).mean())

def alpha110(data, dependencies=['close', 'high', 'low'], max_window=21):
    # SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
    part1 = (np.maximum(data['high']-data['close'].shift(1), 0.0)).rolling(window=20,min_periods=20).sum()
    part2 = (np.maximum(data['close'].shift(1)-data['low'], 0.0)).rolling(window=20,min_periods=20).sum()
    return (part1 / part2) * 100.0

def alpha111(data, dependencies=['low', 'high', 'close', 'volume'], max_window=11):
    # SMA(VOL*(2*CLOSE-LOW-HIGH)/(HIGH-LOW),11,2)-SMA(VOL*(2*CLOSE-LOW-HIGH)/(HIGH-LOW),4,2)
    win_vol = data['volume'] * (data['close']*2-data['low']-data['high']) / (data['high']-data['low'])
    alpha = win_vol.ewm(adjust=False, alpha=float(2)/11, min_periods=0, ignore_na=False).mean() - win_vol.ewm(adjust=False, alpha=float(2)/4, min_periods=0, ignore_na=False).mean()
    return alpha

def alpha112(data, dependencies=['close'], max_window=13):
    # (SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))
    # /(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100
    part1 = (np.maximum(data['close'].diff(1), 0.0)).rolling(window=12, min_periods=12).sum()
    part2 = abs(np.minimum(data['close'].diff(1), 0.0)).rolling(window=12, min_periods=12).sum()
    return ((part1-part2) / (part1+part2)) * 100

def alpha113(data, dependencies=['close', 'volume'], max_window=28):
    # -1*RANK(SUM(DELAY(CLOSE,5),20)/20)*CORR(CLOSE,VOLUME,2)*RANK(CORR(SUM(CLOSE,5),SUM(CLOSE,20),2))
    part1 = (data['close'].shift(5).rolling(window=20, min_periods=20).mean()).rank(axis=0, pct=True)
    part2 = data['close'].rolling(window=2, min_periods=2).corr(data['volume'])
    part3 = ((data['close'].rolling(window=5, min_periods=5).sum()).rolling(window=2, min_periods=2).corr(data['close'].rolling(window=20, min_periods=20).sum())).rank(axis=0, pct=True)
    return (part1 * part2 * part3) * (-1)

def alpha114(data, dependencies=['high', 'low', 'close', 'amount', 'volume'], max_window=8):
    # RANK(DELAY((HIGH-LOW)/(SUM(CLOSE,5)/5),2))*RANK(RANK(VOLUME))/((HIGH-LOW)/(SUM(CLOSE,5)/5)/(VWAP-CLOSE))
    # rank/rank貌似没必要
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    part1 = ((data['high']-data['low'])/(data['close'].rolling(window=5,min_periods=5).mean())).shift(2).rank(axis=0,pct=True)
    part2 = data['volume'].rank(axis=0, pct=True).rank(axis=0, pct=True)
    part3 = (data['high']-data['low'])/(data['close'].rolling(window=5,min_periods=5).mean())/(vwap-data['close'])
    return part1*part2*part3

def alpha115(data, dependencies=['high', 'low', 'volume', 'close'], max_window=40):
    # (RANK(CORR(HIGH*0.9+CLOSE*0.1,MEAN(VOLUME,30),10))^RANK(CORR(TSRANK((HIGH+LOW)/2,4),TSRANK(VOLUME,10),7)))
    part1 = ((data['high'] * 0.9 + data['close'] * 0.1).rolling(window=10, min_periods=10).corr(
        data['volume'].rolling(window=30, min_periods=30).mean())).rank(axis=0, pct=True)
    part2 = (((data['high'] * 0.5 + data['low'] * 0.5).rolling(window=4, min_periods=4).apply(lambda x: stats.rankdata(x)[-1]/4.0)).rolling(window=7, min_periods=7) .corr(data['volume'].rolling(window=10, min_periods=10).apply(lambda x: stats.rankdata(x)[-1]/10.0))).rank(axis=0,pct=True)
    return part1 ** part2
    
def alpha116(data, dependencies=['close'], max_window=20):
    # REGBETA(CLOSE,SEQUENCE,20)
    alpha = REGBETA(data['close'],pd.Series(range(1,21)),20)
    return alpha

def alpha117(data, dependencies=['volume', 'close', 'high', 'low'], max_window=32):
    # TSRANK(VOLUME,32)*(1-TSRANK(CLOSE+HIGH-LOW,16))*(1-TSRANK(RET,32))
    part1 = data['volume'].rolling(32).apply(lambda x:x.rank(pct=True)[-1])
    part2 = 1.0 - (data['close']+data['high']-data['low']).rolling(16).apply(lambda x:x.rank(pct=True)[-1])
    part3 = 1.0 - data['close'].pct_change(periods=1).rolling(32).apply(lambda x:x.rank(pct=True)[-1])
    return part1 * part2 * part3

def alpha118(data, dependencies=['high', 'Open', 'low'], max_window=20):
    # SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
    alpha = (data['high']-data['Open']).rolling(window=20,min_periods=20).sum() / (data['Open']-data['low']).rolling(window=20,min_periods=20).sum() * 100.0
    return alpha

def alpha119(data, dependencies=['amount', 'volume', 'Open'], max_window=62):
    # RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26),5),7))-RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN),RANK(MEAN(VOLUME,15)),21),9),7),8))
    # 感觉有个TSMIN
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    w7 = np.array(range(1, 8))
    w7 = w7/w7.sum()
    w8 = np.array(range(1, 9))
    w8 = w8/w8.sum()
    part1 = ((data['volume'].rolling(window=5,min_periods=5).mean()).rolling(window=26, min_periods=26).sum()).rolling(window=5, min_periods=5).corr(vwap)
    part1 = (part1.rolling(window=7,min_periods=7).apply(lambda x:np.dot(x,w7))).rank(axis=0,pct=True)
    part2 = ((data['volume'].rolling(window=15, min_periods=15).mean()).rank(axis=0,pct=True)).rolling(window=21,min_periods=21).corr(data['Open'].rank(axis=0,pct=True))
    part2 = (((part2.rolling(window=9, min_periods=9).min()).rolling(window=7,min_periods=7).apply(lambda x: stats.rankdata(x)[-1]/7.0)).rolling(window=8,min_periods=8).apply(lambda x:np.dot(x,w8))).rank(axis=0, pct=True)
    return part1-part2

def alpha120(data, dependencies=['amount', 'volume', 'close'], max_window=1):
    # RANK(VWAP-CLOSE)/RANK(VWAP+CLOSE)
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    return ((vwap-data['close']) / (vwap+data['close']))

def alpha121(data, dependencies=['amount', 'volume'], max_window=83):
    # (RANK(VWAP-MIN(VWAP,12))^TSRANK(CORR(TSRANK(VWAP,20),TSRANK(MEAN(VOLUME,60),2),18),3))*-1
    # vwap = data['amount'] / (data['volume']*100) 
    vwap = data['vwap']
    part1 = (vwap - vwap.rolling(window=12, min_periods=12).min()).rank(axis=0, pct=True)
    part2 = (data['volume'].rolling(window=60, min_periods=60).mean()).rolling(window=2, min_periods=2).apply(lambda x: stats.rankdata(x)[-1]/2.0)
    part2 = ((vwap.rolling(window=20, min_periods=20).apply(lambda x: stats.rankdata(x)[-1]/20.0)).rolling(window=18, min_periods=18).corr(part2)) .rolling(window=3, min_periods=3).apply(lambda x: stats.rankdata(x)[-1]/3.0)
    return (part1 ** part2) * (-1)

def alpha122(data, dependencies=['close'], max_window=40):
    # (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
    part1 = (np.log(data['close'])).ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean()
    part1 = (part1.ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean()).ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean()
    return part1.pct_change(periods=1)

def alpha123(data, dependencies=['high', 'low', 'volume'], max_window=89):
    # (RANK(CORR(SUM((HIGH+LOW)/2,20),SUM(MEAN(VOLUME,60),20),9)) < RANK(CORR(LOW,VOLUME,6)))*-1
    part1 = (data['high']*0.5+data['low']*0.5).rolling(window=20, min_periods=20).sum()
    part1 = ((data['volume'].rolling(window=60,min_periods=60).mean()).rolling(window=20,min_periods=20).sum()).rolling(window=9,min_periods=9).corr(part1).rank(axis=0, pct=True)
    part2 = (data['low'].rolling(window=6,min_periods=6).corr(data['volume'])).rank(axis=0, pct=True)
    return (part2 - part1) * (-1)

def alpha124(data, dependencies=['close', 'amount', 'volume'], max_window=32):
    # (CLOSE-VWAP)/DECAYLINEAR(RANK(TSMAX(CLOSE,30)),2)
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    w2 = np.array(range(1, 3))
    w2 = w2/w2.sum()
    part1 = data['close'] - vwap
    part2 = ((data['close'].rolling(window=30,min_periods=30).max()).rank(axis=0,pct=True)).rolling(window=2,min_periods=2).apply(lambda x:np.dot(x,w2))
    return part1 / part2

def alpha125(data, dependencies=['close', 'amount', 'volume'], max_window=117):
    # RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,80),17),20))/RANK(DECAYLINEAR(DELTA(CLOSE*0.5+VWAP*0.5,3),16))
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    w20 = np.array(range(1, 21))
    w20 = w20/w20.sum()
    w16 = np.array(range(1, 17))
    w16 = w16/w16.sum()
    part1 = (data['volume'].rolling(window=80,min_periods=80).mean()).rolling(window=17,min_periods=17).corr(vwap)
    part1 = (part1.rolling(window=20,min_periods=20).apply(lambda x:np.dot(x,w20))).rank(axis=0, pct=True)
    part2 = ((data['close']*0.5+vwap*0.5).diff(periods=3)).rolling(window=16,min_periods=16).apply(lambda x:np.dot(x,w16)).rank(axis=0,pct=True)
    return part1 / part2

def alpha126(data, dependencies=['high', 'low', 'close'], max_window=1):
    # (CLOSE+HIGH+LOW)/3
    return (data['close'] + data['high'] + data['low']) / 3.0

def alpha127(data, dependencies=['close'], max_window=24):
    # MEAN((100*(CLOSE-MAX(CLOSE,12))/MAX(CLOSE,12))^2)^(1/2)
    # 这里貌似是TSMAX,MEAN少一个参数
    alpha = (data['close'] - data['close'].rolling(window=12,min_periods=12).max()) / data['close'].rolling(window=12,min_periods=12).max() * 100
    alpha = (alpha ** 2).rolling(window=12, min_periods=12).mean() ** 0.5
    return alpha

def alpha128(data, dependencies=['high', 'low', 'close', 'volume'], max_window=14):
    # 100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/
    # SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)))
    condition1 = ((data['high']+data['low']+data['close'])/3.0).diff(1) > 0.0
    condition2 = ((data['high']+data['low']+data['close'])/3.0).diff(1) < 0.0
    part1 = (data['high']+data['low']+data['close'])/3.0*data['volume']
    part2 = part1.copy(deep=True)
    part1[~condition1] = 0.0
    part1 = part1.rolling(window=14, min_periods=14).sum()
    part2[~condition2] = 0.0
    part2 = part2.rolling(window=14, min_periods=14).sum()
    return (100.0-(100.0/(1+part1/part2)))

def alpha129(data, dependencies=['close'], max_window=13):
    # SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)
    return (abs(np.minimum(data['close'].diff(1), 0.0))).rolling(window=12, min_periods=12).sum()

def alpha130(data, dependencies=['low', 'high', 'volume', 'amount'], max_window=59):
    # (RANK(DECAYLINEAR(CORR((HIGH+LOW)/2,MEAN(VOLUME,40),9),10))/RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),7),3)))
    # vwap = data['amount'] / (data['volume']*100) 
    vwap = data['vwap']
    w10 = np.array(range(1, 11))
    w10 = w10/w10.sum()
    w3 = np.array(range(1, 4))
    w3 = w3/w3.sum()
    part1 = (data['volume'].rolling(window=40,min_periods=40).mean()).rolling(window=9,min_periods=9).corr(data['high']*0.5+data['low']*0.5)
    part1 = part1.rolling(window=10,min_periods=10).apply(lambda x: np.dot(x, w10)).rank(axis=0, pct=True)
    part2 = (data['volume'].rank(axis=0, pct=True)).rolling(window=7,min_periods=7).corr(vwap.rank(axis=0, pct=True))
    part2 = part2.rolling(window=3,min_periods=3).apply(lambda x: np.dot(x, w3)).rank(axis=0, pct=True)
    return part1 / part2

def alpha131(data, dependencies=['amount', 'volume', 'close'], max_window=86):
    # (RANK(DELAT(VWAP,1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50),18),18))
    # vwap = data['amount'] / (data['volume']*100) 
    vwap = data['vwap']
    part1 = vwap.diff(1).rank(axis=0, pct=True)
    part2 = (data['volume'].rolling(window=50, min_periods=50).mean()).rolling(window=18, min_periods=18).corr(data['close'])
    part2 = part2.rolling(window=18, min_periods=18).apply(lambda x:x.rank(pct=True)[-1])
    return (part1 ** part2)

def alpha132(data, dependencies=['amount'], max_window=20):
    # MEAN(AMOUNT,20)
    return data['amount'].rolling(window=20, min_periods=20).mean()

def alpha133(data, dependencies=['low', 'high'], max_window=20):
    # ((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100
    part1 = (20 - data['high'].rolling(window=20, min_periods=20).apply(lambda x: 19-x.argmax(axis=0))) * 5.0
    part2 = (20 - data['low'].rolling(window=20, min_periods=20).apply(lambda x: 19-x.argmin(axis=0))) * 5.0
    return part1 -part2

def alpha134(data, dependencies=['close', 'volume'], max_window=13):
    # (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
    return (data['close'].pct_change(periods=12) * data['volume'])

def alpha135(data, dependencies=['close'], max_window=42):
    # SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
    alpha = (data['close']/data['close'].shift(20)).shift(1)
    return alpha.ewm(adjust=False, alpha=float(1)/20, min_periods=0, ignore_na=False).mean()

def alpha136(data, dependencies=['close', 'Open', 'volume'], max_window=10):
    # -1*RANK(DELTA(RET,3))*CORR(OPEN,VOLUME,10)
    part1 = data['close'].pct_change(periods=1).diff(3).rank(axis=0,pct=True)
    part2 = data['Open'].rolling(window=10, min_periods=10).corr(data['volume'])
    return (part1 * part2) * (-1)

def alpha137(data, dependencies=['Open', 'low', 'close', 'high'], max_window=2):
    # 16*(CLOSE+(CLOSE-OPEN)/2-DELAY(OPEN,1))/
    # ((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1))&ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:
    # (ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))
    # *MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))
    part1 = data['close'] * 1.5 - data['Open'] * 0.5 - data['Open'].shift(1)
    part2 = abs(data['high']-data['close'].shift(1)) + abs(data['low']-data['close'].shift(1)) / 2.0 + abs(data['close']-data['Open']).shift(1) / 4.0
    condition1 = np.logical_and(abs(data['high']-data['close'].shift(1)) > abs(data['low']-data['close'].shift(1)), 
                               abs(data['high']-data['close'].shift(1)) > abs(data['high']-data['low'].shift(1)))
    condition2 = np.logical_and(abs(data['low']-data['close'].shift(1)) > abs(data['high']-data['low'].shift(1)), 
                               abs(data['low']-data['close'].shift(1)) > abs(data['high']-data['close'].shift(1)))
    part2[~condition1 & condition2] = abs(data['low']-data['close'].shift(1)) + abs(data['high']-data['close'].shift(1)) / 2.0 + abs(data['close']-data['Open']).shift(1) / 4.0
    part2[~condition1 & ~condition2] = abs(data['high']-data['low'].shift(1)) + abs(data['close']-data['Open']).shift(1) / 4.0
    part3 = np.maximum(abs(data['high']-data['close'].shift(1)), abs(data['low']-data['close'].shift(1)))
    alpha = (part1 / part2 * part3 * 16.0)
    return alpha

def alpha138(data, dependencies=['low','amount','volume'], max_window=126):
    # ((RANK(DECAYLINEAR(DELTA(LOW*0.7+VWAP*0.3,3),20))
    # -TSRANK(DECAYLINEAR(TSRANK(
        # CORR(TSRANK(LOW,8),TSRANK(MEAN(VOLUME,60),17),5)
        # ,19),16),7))* -1)
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    w20 = np.array(range(1, 21))
    w20 = w20/w20.sum()
    w16 = np.array(range(1, 17))
    w16 = w16/w16.sum()
    part1 = ((data['low']*0.7+vwap*0.3).diff(3)).rolling(window=20,min_periods=20).apply(lambda x: np.dot(x,w20)).rank(axis=0, pct=True)
    part2 = (data['volume'].rolling(window=60, min_periods=60).mean()).rolling(window=17,min_periods=17).apply(lambda x: stats.rankdata(x)[-1]/17.0)
    part2 = part2.rolling(window=5,min_periods=5).corr(data['low'].rolling(window=8,min_periods=8).apply(lambda x: stats.rankdata(x)[-1]/8.0))
    part2 = ((part2.rolling(window=19,min_periods=19).apply(lambda x: stats.rankdata(x)[-1]/19.0)).rolling(window=16,min_periods=16).apply(lambda x:np.dot(x,w16))).rolling(window=7,min_periods=7).apply(lambda x: stats.rankdata(x)[-1]/7.0)
    return (part1-part2) * (-1)

def alpha139(data, dependencies=['Open', 'volume'], max_window=10):
    # (-1*CORR(OPEN,VOLUME,10))
    return data['Open'].rolling(window=10,min_periods=10).corr(data['volume']) * (-1)

def alpha140(data, dependencies=['Open', 'low', 'high', 'close', 'volume'], max_window=99):
    # MIN(RANK(DECAYLINEAR(RANK(OPEN)+RANK(LOW)-RANK(HIGH)-RANK(CLOSE),8)),TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE,8),TSRANK(MEAN(VOLUME,60),20),8),7),3))
    w8 = np.array(range(1, 9))
    w8 = w8/w8.sum()
    w7 = np.array(range(1, 8))
    w7 = w7/w7.sum()
    part1 = data['Open'].rank(axis=0,pct=True)+data['low'].rank(axis=0,pct=True)-data['high'].rank(axis=0,pct=True)-data['close'].rank(axis=0,pct=True)
    part1 = part1.rolling(window=8,min_periods=8).apply(lambda x:np.dot(x,w8)).rank(axis=0,pct=True)
    part2 = (data['volume'].rolling(window=60, min_periods=60).mean()).rolling(window=20,min_periods=20).apply(lambda x: stats.rankdata(x)[-1]/20.0)
    part2 = part2.rolling(window=8,min_periods=8).corr(data['close'].rolling(window=8,min_periods=8).apply(lambda x: stats.rankdata(x)[-1]/8.0))
    part2 = (part2.rolling(window=7,min_periods=7).apply(lambda x:np.dot(x,w7))).rolling(window=3,min_periods=3).apply(lambda x: stats.rankdata(x)[-1]/3.0)  
    return np.minimum(part1,part2)

def alpha141(data, dependencies=['high', 'volume'], max_window=25):
    # (RANK(CORR(RANK(HIGH),RANK(MEAN(VOLUME,15)),9))*-1)
    alpha = ((data['volume'].rolling(window=15,min_periods=15).mean().rank(axis=0,pct=True)).rolling(window=9,min_periods=9).corr(data['high'].rank(axis=0,pct=True))).rank(axis=0,pct=True)
    return alpha * (-1)

def alpha142(data, dependencies=['close', 'volume'], max_window=25):
    # -1*RANK(TSRANK(CLOSE,10))*RANK(DELTA(DELTA(CLOSE,1),1))*RANK(TSRANK(VOLUME/MEAN(VOLUME,20),5))
    part1 = (data['close'].rolling(window=10,min_periods=10).apply(lambda x: stats.rankdata(x)[-1]/10.0)).rank(axis=0,pct=True)
    part2 = (data['close'].diff(1)).diff(1).rank(axis=0,pct=True)
    part3 = (data['volume']/data['volume'].rolling(window=20,min_periods=20).mean()).rolling(window=5,min_periods=5).apply(lambda x: stats.rankdata(x)[-1]/5.0).rank(axis=0,pct=True)
    return (part1 * part2 * part3) * (-1)

def alpha143(data):
    # CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF
    condition = data['close']>data['close'].shift(1)
    alpha = data['close'].copy()
    alpha[condition] = ((1+data['close'].pct_change())*data['close'])[condition]
    return alpha

def alpha144(data, dependencies=['close','amount'], max_window=21):
    # SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)
    part1 = abs(data['close'].pct_change(periods=1)) / data['amount']
    part1[data['close'].diff(1)>=0] = 0.0
    part1 = part1.rolling(window=20, min_periods=20).sum()
    part2 = (data['close'].diff(1)<0.0).rolling(window=20,min_periods=20).sum()
    return part1 / part2

def alpha145(data, dependencies=['volume'], max_window=26):
    # (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100
    alpha = (data['volume'].rolling(window=9,min_periods=9).mean() - data['volume'].rolling(window=26,min_periods=26).mean()) / data['volume'].rolling(window=12,min_periods=12).mean() * 100.0
    return alpha

def alpha146(data, dependencies=['close'], max_window=121):
    # MEAN(RET-SMA(RET,61,2),20)*(RET-SMA(RET,61,2))/SMA(SMA(RET,61,2)^2,60)
    # 假设最后一个SMA(X,60,1)
    sma = (data['close'].pct_change(1)).ewm(adjust=False, alpha=float(2)/61, min_periods=0, ignore_na=False).mean()
    ret_excess = data['close'].pct_change(1) - sma
    part1 = ret_excess.rolling(window=20, min_periods=20).mean() * ret_excess
    part2 = (sma ** 2).ewm(adjust=False, alpha=float(1)/60, min_periods=0, ignore_na=False).mean()
    return part1 / part2


def alpha147(data, dependencies=['close'], max_window=24):
    # REGBETA(MEAN(CLOSE,12),SEQUENCE(12))
    ma_price = data['close'].rolling(window=12, min_periods=12).mean()
    alpha = REGBETA(ma_price,pd.Series(range(1,13)),12)
    return alpha

def alpha148(data, dependencies=['Open', 'volume'], max_window=75):
    # (RANK(CORR(OPEN,SUM(MEAN(VOLUME,60),9),6))<RANK(OPEN-TSMIN(OPEN,14)))*-1
    part1 = (data['volume'].rolling(window=60,min_periods=60).mean()).rolling(window=9,min_periods=9).sum()
    part1 = part1.rolling(window=6,min_periods=6).corr(data['Open']).rank(axis=0,pct=True)
    part2 = (data['Open'] - data['Open'].rolling(window=14,min_periods=14).min()).rank(axis=0, pct=True)
    return (part2-part1) * (-1)

# def alpha149(data, dependencies=['close'], max_window=253):
#     # REGBETA(FILTER(RET,BANCHMARK_INDEX_CLOSE<DELAY(BANCHMARK_INDEX_CLOSE,1)),
#     # FILTER(BANCHMARK_INDEX_CLOSE/DELAY(BANCHMARK_INDEX_CLOSE,1)-1,BANCHMARK_INDEX_CLOSE<DELAY(BANCHMARK_INDEX_CLOSE,1)),252)
#     bm = (data['close'].mean(axis=0).diff(1) < 0.0)
#     part1 = data['close'].pct_change(periods=1).iloc[-252:][bm]
#     part2 = data['close'].mean(axis=0).pct_change(periods=1).iloc[-252:][bm]
#     alpha = pd.DataFrame([[stats.linregress(part1[col].values, part2.values)[0] for col in data['close'].columns]], 
#                  index=data['close'].index[-1:], columns=data['close'].columns)
#     return alpha
def alpha149(data, dependencies=['close', 'bm_index_close'], max_window=253):
    # REGBETA(FILTER(RET,BANCHMARK_INDEX_CLOSE<DELAY(BANCHMARK_INDEX_CLOSE,1)),
    # FILTER(BANCHMARK_INDEX_CLOSE/DELAY(BANCHMARK_INDEX_CLOSE,1)-1,BANCHMARK_INDEX_CLOSE<DELAY(BANCHMARK_INDEX_CLOSE,1)),252)
    bm = data['bm_index_close']
    bm = (bm.diff(1) < 0.0)
    part1 = data['close'].pct_change(periods=1)[bm]
    part2 = data['close'].rolling(252).mean().pct_change(periods=1)[bm]
    try:
        alpha = REGBETA(part1,part2,252)
    except Exception as e:
        print(e)
        return pd.Series(np.nan,index=data.index)
    return alpha


def alpha150(data, dependencies=['close', 'high', 'low', 'volume'], max_window=1):
    # (CLOSE+HIGH+LOW)/3*VOLUME
    return ((data['close'] + data['high'] + data['low']) / 3.0 * data['volume'])

def alpha151(data, dependencies=['close'], max_window=41):
    # SMA(CLOSE-DELAY(CLOSE,20),20,1)
    return (data['close'].diff(20)).ewm(adjust=False, alpha=float(1)/20, min_periods=0, ignore_na=False).mean()

def alpha152(data, dependencies=['close'], max_window=59):
    # A=DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1)
    # SMA(MEAN(A,12)-MEAN(A,26),9,1)
    part1 = ((data['close'] / data['close'].shift(9)).shift(1)).ewm(adjust=False, alpha=float(1)/9, min_periods=0, ignore_na=False).mean().shift(1)
    alpha = (part1.rolling(window=12,min_periods=12).mean()-part1.rolling(window=26,min_periods=26).mean()).ewm(adjust=False, alpha=float(1)/9, min_periods=0, ignore_na=False).mean()
    return alpha

def alpha153(data, dependencies=['BBI'], max_window=24):
    # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
    # 就是BBI
    part1=[3,6,12,24]
    part2=[data['close'].rolling(window=x,min_periods=x).mean() for x in part1]
    alpha = sum(part2)/4
    return alpha

def alpha154(data, dependencies=['amount', 'volume'], max_window=198):
    # VWAP-MIN(VWAP,16)<CORR(VWAP,MEAN(VOLUME,180),18)
    # 感觉是TSMIN
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    part1 = vwap - vwap.rolling(window=16, min_periods=16).min()
    part2 = (data['volume'].rolling(window=180, min_periods=180).mean()).rolling(window=18, min_periods=18).corr(vwap)
    return part2-part1

def alpha155(data, dependencies=['volume'], max_window=37):
    # SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
    sma13 = data['volume'].ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean()
    sma27 = data['volume'].ewm(adjust=False, alpha=float(2)/27, min_periods=0, ignore_na=False).mean()
    ssma = (sma13-sma27).ewm(adjust=False, alpha=float(2)/10, min_periods=0, ignore_na=False).mean()
    return (sma13 - sma27 - ssma)

def alpha156(data, dependencies=['amount', 'volume', 'Open', 'low'], max_window=9):
    # MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),RANK(DECAYLINEAR((DELTA(OPEN*0.15+LOW*0.85,2)/(OPEN*0.15+LOW*0.85)) * -1,3))) * -1
    w3 = np.array(range(1, 4))
    w3 = w3/w3.sum()
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    den = data['Open']*0.15+data['low']*0.85
    part1 = (vwap.diff(5)).rolling(window=3,min_periods=3).apply(lambda x:np.dot(x,w3))
    part2 = (den.diff(2)/den*(-1)).rolling(window=3,min_periods=3).apply(lambda x:np.dot(x,w3))
    return np.maximum(part1, part2) * (-1)

def alpha157(data, dependencies=['close'], max_window=12):
    # MIN(PROD(RANK(LOG(SUM(TSMIN(RANK(-1*RANK(DELTA(CLOSE-1,5))),2),1))),1),5) +TSRANK(DELAY(-1*RET,6),5)
    part1 = np.log((((data['close']-1.0).diff(5).rank(pct=True) * (-1)).rank(pct=True)).rolling(window=2, min_periods=2).min())
    part1 = (part1.rank(axis=0, pct=True)).rolling(window=5,min_periods=5).min()
    part2 = ((data['close'].pct_change(periods=1) * (-1)).shift(6)).rolling(5).apply(lambda x:x.rank(pct=True)[-1])
    return (part1 + part2)

def alpha158(data, dependencies=['low', 'high', 'close'], max_window=1):
    # (HIGH-LOW)/CLOSE
    return ((data['high'] - data['low']) / data['close'])

def alpha159(data, dependencies=['close', 'low', 'high'], max_window=25):
    # ((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24
    # +(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24
    # +(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)
    min_low_close = np.minimum(data['low'], data['close'].shift(1))
    max_high_close = np.maximum(data['high'], data['close'].shift(1))
    part1 = (data['close'] - min_low_close.rolling(window=6,min_periods=6).sum()) / (max_high_close-min_low_close).rolling(window=6,min_periods=6).sum() * 12 * 24
    part2 = (data['close'] - min_low_close.rolling(window=12,min_periods=12).sum()) / (max_high_close-min_low_close).rolling(window=12,min_periods=12).sum() * 6 * 24
    part3 = (data['close'] - min_low_close.rolling(window=24,min_periods=24).sum()) / (max_high_close-min_low_close).rolling(window=24,min_periods=24).sum() * 6 * 12
    return (part1+part2+part3)*100.0/(12*6+6*24+12*24)
    
def alpha160(data, dependencies=['close'], max_window=41):
    # SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    part1 = data['close'].rolling(window=20,min_periods=20).std()
    part1[data['close'].diff(1)>0] = 0.0
    part1[:19] = np.nan
    return part1.ewm(adjust=False, alpha=float(1)/20, min_periods=0, ignore_na=False).mean()

def alpha161(data, dependencies=['close', 'low', 'high'], max_window=13):
    # MEAN(MAX(MAX(HIGH-LOW,ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
    part1 = np.maximum(data['high']-data['low'], abs(data['close'].shift(1)-data['high']))
    part1 = np.maximum(part1, abs(data['close'].shift(1)-data['low']))
    return part1.rolling(window=12,min_periods=12).mean()

def alpha162(data, dependencies=['close'], max_window=25):
    # (SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
    # -MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))
    # /(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)
    # -MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))
    den = (np.maximum(data['close'].diff(1), 0.0)).ewm(adjust=False, alpha=float(1)/12, min_periods=0, ignore_na=False).mean() /(abs(data['close'].diff(1))).ewm(adjust=False, alpha=float(1)/12, min_periods=0, ignore_na=False).mean() * 100.0
    alpha = (den - den.rolling(window=12,min_periods=12).min()) / (den.rolling(window=12,min_periods=12).max() - den.rolling(window=12,min_periods=12).min())
    return alpha

def alpha163(data, dependencies=['amount', 'volume', 'close', 'high'], max_window=20):
    # RANK((-1*RET)*MEAN(VOLUME,20)*VWAP*(HIGH-CLOSE))
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    alpha = data['close'].pct_change(periods=1) * (data['volume'].rolling(window=20, min_periods=20).mean()) * vwap * (data['high'] - data['close']) * (-1)
    return alpha

def alpha164(data, dependencies=['close', 'high', 'low'], max_window=26):
    # SMA(((CLOSE>DELAY(CLOSE,1)?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(CLOSE>DELAY(CLOSE,1)?1/(CLOSE-DELAY(CLOSE,1)):1,12))/(HIGH-LOW)*100,13,2)
    part1 = 1.0 / data['close'].diff(1)
    part1[data['close'].diff(1)<=0] = 1.0
    part2 = part1.rolling(window=12, min_periods=12).min()
    alpha = (part1-part2)/(data['high']-data['low'])*100.0
    return alpha.ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean()

def alpha165(data, dependencies=['close'], max_window=144):
    # MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)
    # SUMAC少了前N项和,TSMAX/TSMIN
    part1 = ((data['close']-data['close'].rolling(window=48,min_periods=48).mean()).rolling(window=48,min_periods=48).sum()).rolling(window=48,min_periods=48).max()
    part2 = ((data['close']-data['close'].rolling(window=48,min_periods=48).mean()).rolling(window=48,min_periods=48).sum()).rolling(window=48,min_periods=48).min()
    part3 = data['close'].rolling(window=48,min_periods=48).std()
    return (part1-part2/part3)

def alpha166(data, dependencies=['close'], max_window=41):
    # -20*(20-1)^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)*(SUM((CLOSE/DELAY(CLOSE,1))^2,20))^1.5)
    part1 = data['close'].pct_change(periods=1)-(data['close'].pct_change(periods=1).rolling(window=20,min_periods=20).mean())
    part1 = part1.rolling(window=20,min_periods=20).sum() * ((-20) * 19 ** 1.5)
    part2 = (((data['close']/data['close'].shift(1)) ** 2).rolling(window=20,min_periods=20).sum() ** 1.5) * 19 * 18
    return (part1 / part2)

def alpha167(data, dependencies=['close'], max_window=13):
    # SUM(CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0,12)
    return (np.maximum(data['close'].diff(1), 0.0)).rolling(window=12, min_periods=12).sum()

def alpha168(data, dependencies=['volume'], max_window=20):
    # -1*VOLUME/MEAN(VOLUME,20)
    return (data['volume']/(data['volume'].rolling(window=20,min_periods=20).mean())) * (-1)

def alpha169(data, dependencies=['close'], max_window=48):
    # SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)
    part1 = (data['close'].diff(1).ewm(adjust=False, alpha=float(1)/9, min_periods=0, ignore_na=False).mean()).shift(1)
    part2 = (part1.rolling(window=12, min_periods=12).mean() - part1.rolling(window=26, min_periods=26).mean()).ewm(adjust=False, alpha=float(1)/10, min_periods=0, ignore_na=False).mean()
    return part2

def alpha170(data, dependencies=['close','volume','high', 'amount'], max_window=20):
    # ((RANK(1/CLOSE)*VOLUME)/MEAN(VOLUME,20))*(HIGH*RANK(HIGH-CLOSE)/(SUM(HIGH,5)/5))-RANK(VWAP-DELAY(VWAP,5))
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    part1 = (1.0/data['close']).rank(axis=0,pct=True) * data['volume'] / (data['volume'].rolling(window=20,min_periods=20).mean())
    part2 = ((data['high']-data['close']).rank(axis=0,pct=True) * data['high']) / (data['high'].rolling(window=5,min_periods=5).sum()/5.0)
    part3 = (vwap.diff(5)).rank(axis=0,pct=True)
    return (part1*part2-part3)
    
def alpha171(data, dependencies=['low', 'close', 'Open', 'high'], max_window=1):
    # (-1*(LOW-CLOSE)*(OPEN^5))/((CLOSE-HIGH)*(CLOSE^5))
    part1 = (data['low']-data['close']) * (data['Open'] ** 5) * (-1)
    part2 = (data['close']-data['high']) * (data['close'] ** 5)
    return round(part1/part2,8)

def alpha172(data, dependencies=['ADX'], max_window=20):
    # 就是DMI-ADX
    # HD  HIGH-DELAY(HIGH,1)
    # LD  DELAY(LOW,1)-LOW
    # TR  MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))

    # MEAN(ABS(
    #     SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)
    #     -SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))
    # /(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)
    #     +SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))
    # *100,6)
    hd=data['high'].diff(1)
    ld=-data['low'].diff(1)
    tr=np.maximum(np.maximum(data['high']-data['low'],(data['high']-data['close'].shift(1)).abs()),(data['low']-data['close'].shift(1)).abs())
    part1=(((ld>0)&(ld>hd))*ld).rolling(window=14, min_periods=14).sum()*100/tr.rolling(window=14, min_periods=14).sum()-(((hd>0)&(hd>ld))*hd).rolling(window=14, min_periods=14).sum()*100/tr.rolling(window=14, min_periods=14).sum()
    part2=(((ld>0)&(ld>hd))*ld).rolling(window=14, min_periods=14).sum()*100/tr.rolling(window=14, min_periods=14).sum()+(((hd>0)&(hd>ld))*hd).rolling(window=14, min_periods=14).sum()*100/tr.rolling(window=14, min_periods=14).sum()
    alpha = (part1/part2).abs().rolling(6).mean()*100
    return alpha

def alpha173(data, dependencies=['close'], max_window=39):
    # 3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)
    den = data['close'].ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean()
    part1 = 3 * den
    part2 = 2 * (den.ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean())
    part3 = ((np.log(data['close']).ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean()) .ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean()) .ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean()
    return part1-part2+part3

def alpha174(data, dependencies=['close'], max_window=41):
    # SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    part1 = data['close'].rolling(window=20,min_periods=20).std()
    part1[data['close'].diff(1)<=0] = 0.0
    part1[:19] = np.nan
    return part1.ewm(adjust=False, alpha=float(1)/20, min_periods=0, ignore_na=False).mean()

def alpha175(data, dependencies=['low','high','close'], max_window=7):
    # MEAN(MAX(MAX(HIGH-LOW,ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
    alpha = np.maximum(data['high']-data['low'], abs(data['close'].shift(1)-data['high']))
    alpha = np.maximum(alpha, abs(data['close'].shift(1)-data['low']))
    return alpha.rolling(window=6,min_periods=6).mean()

def alpha176(data, dependencies=['close','high','low','volume'], max_window=18):
    # CORR(RANK((CLOSE-TSMIN(LOW,12))/(TSMAX(HIGH,12)-TSMIN(LOW,12))),RANK(VOLUME),6)
    part1 = ((data['close'] - data['low'].rolling(window=12,min_periods=12).min()) / (data['high'].rolling(window=12,min_periods=12).max()-data['low'].rolling(window=12,min_periods=12).min())).rank(axis=0, pct=True)
    part2 = data['volume'].rank(axis=0, pct=True)
    return part1.rolling(window=6,min_periods=6).corr(part2)

def alpha177(data, dependencies=['high'], max_window=20):
    # ((20-HIGHDAY(HIGH,20))/20)*100
    return (20 - data['high'].rolling(window=20, min_periods=20).apply(lambda x: 19-x.argmax(axis=0))) * 5.0

def alpha178(data, dependencies=['close', 'volume'], max_window=2):
    # (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
    return (data['close'].pct_change(periods=1) * data['volume'])

def alpha179(data, dependencies=['low','amount','volume'], max_window=62):
    # RANK(CORR(VWAP,VOLUME,4))*RANK(CORR(RANK(LOW),RANK(MEAN(VOLUME,50)),12))
    part1 = ((data['amount']/data['volume']).rolling(window=4,min_periods=4).corr(data['volume'])).rank(axis=0,pct=True)
    part2 = (((data['volume'].rolling(window=50,min_periods=50).mean()).rank(axis=0,pct=True)).rolling(window=12,min_periods=12).corr(data['low'].rank(axis=0,pct=True))).rank(axis=0,pct=True)
    return part1 * part2

def alpha180(data, dependencies=['volume', 'close'], max_window=68):
    # (MEAN(VOLUME,20)<VOLUME)?((-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7)):(-1*VOLUME))
    condition = data['volume'].rolling(window=20, min_periods=20).mean() < data['volume']
    alpha = abs(data['close'].diff(7)).rolling(window=60, min_periods=60).apply(lambda x: stats.rankdata(x)[-1]/60.0) * np.sign(data['close'].diff(7)) * (-1)
    alpha[~condition] = -1 * data['volume'][~condition]
    return round(alpha, 8)


def alpha181(data, dependencies=['close', 'bm_index_close'], max_window=40):
    # SUM(RET-MEAN(RET,20)-(BANCHMARK_INDEX_CLOSE-MEAN(BANCHMARK_INDEX_CLOSE,20))^2,20)/SUM((BANCHMARK_INDEX_CLOSE-MEAN(BANCHMARK_INDEX_CLOSE,20))^3)
    # 优化：数值取对数，否则ret 跟 index 可能不在一个量级上，导致全部结果趋同。
    bm = np.log(data['bm_index_close'])
    bm_mean = bm - bm.rolling(window=20, min_periods=20).mean()
    # print(bm_mean)
    # bm_mean = pd.DataFrame(data=np.repeat(bm_mean.values.reshape(len(bm_mean.values),1), len(data['close'].columns), axis=1), index=data['close'].index, columns=data['close'].columns)
    ret = np.log(data['close']).pct_change(periods=1)
    part1 = (ret-ret.rolling(window=20,min_periods=20).mean()-bm_mean**2).rolling(window=20,min_periods=20).sum()
    part2 = (bm_mean ** 3).rolling(window=20,min_periods=20).sum()
    return part1 / part2

def alpha182(data, dependencies=['close','Open', 'bm_index_open', 'bm_index_close'], max_window=20):
    # COUNT((CLOSE>OPEN & BANCHMARK_INDEX_CLOSE>BANCHMARK_INDEX_OPEN) OR (CLOSE<OPEN &BANCHMARK_INDEX_CLOSE<BANCHMARK_INDEX_OPEN),20)/20
    bm = data['bm_index_close'] > data['bm_index_open']
    bm_2 = data['bm_index_close'] < data['bm_index_open']
    # bm = pd.DataFrame(data=np.repeat(bm.values.reshape(len(bm.values),1), len(data['close'].columns), axis=1), index=data['close'].index, columns=data['close'].columns)
    condition_1 = np.logical_and(data['close']>data['Open'], bm)
    condition_2 = np.logical_and(data['close']<data['Open'], bm_2)
    alpha = np.logical_or(condition_1, condition_2).rolling(window=20, min_periods=20).mean()
    return alpha
    
def alpha183(data, dependencies=['close'], max_window=72):
    # MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)
    part1 = ((data['close']-data['close'].rolling(window=24,min_periods=24).mean()).rolling(window=24,min_periods=24).sum()).rolling(window=24,min_periods=24).max()
    part2 = ((data['close']-data['close'].rolling(window=24,min_periods=24).mean()).rolling(window=24,min_periods=24).sum()).rolling(window=24,min_periods=24).min()
    part3 = data['close'].rolling(window=24,min_periods=24).std()
    return part1-part2/part3

def alpha184(data, dependencies=['close','Open'], max_window=201):
    # RANK(CORR(DELAY(OPEN-CLOSE,1),CLOSE,200))+RANK(OPEN-CLOSE)
    part1 = (((data['Open']-data['close']).shift(1)).rolling(window=200,min_periods=200).corr(data['close'])).rank(axis=0,pct=True)
    part2 = (data['Open']-data['close']).rank(axis=0,pct=True)
    return part1+part2

def alpha185(data, dependencies=['close', 'Open'], max_window=1):
    # RANK(-1*(1-OPEN/CLOSE)^2)
    return ((1.0-data['Open']/data['close']) ** 2) * (-1)

def alpha186(data, dependencies=['low','high','close'], max_window=20):
    # 就是ADXR
#     (MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14) - SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14)) 
#     /
#     (SUM((LD>0  &  LD>HD)?LD:0,14)*100/SUM(TR,14) + SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)
#     +
#     DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14) - SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14)) 
#            / (SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14) + SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)
#          ,6))/2
    dm_plus = data['close'].diff(1).fillna(0)
    dm_subtract = data['low'].diff(1).fillna(0)
    condition_plus = (dm_plus<dm_subtract) | (dm_plus<0)
    condition_sub = (dm_subtract < dm_plus) | (dm_subtract<0)
    dm_plus[condition_plus] = 0
    dm_subtract[condition_sub] = 0
    
    close_delay = data['close'].shift(1)
    tr_a = data['high'] - data['low']
    tr_b = data['high'] - close_delay
    tr_c = data['low'] - close_delay
    tr = pd.concat([tr_a,tr_b,tr_c],axis=1).max(axis=1)
    
    tr_sum = tr.rolling(14).sum()
    
    PDI = dm_plus.rolling(14).sum() * 100 / tr_sum
    MDI = dm_subtract.rolling(14).sum() * 100 / tr_sum
    DX = np.abs(PDI-MDI)/(PDI+MDI) * 100
    ADX = MEAN(DX,6)
    ADXR =(ADX+ADX.shift(1))/2
    
    return ADXR



def alpha187(data, dependencies=['Open', 'high'], max_window=21):
    # SUM(OPEN<=DELAY(OPEN,1)?0:MAX(HIGH-OPEN,OPEN-DELAY(OPEN,1)),20)
    part1 = np.maximum(data['high']-data['Open'], data['Open'].diff(1))
    part1[data['Open'].diff(1)<=0] = 0.0
    return part1.rolling(window=20, min_periods=20).sum()

def alpha188(data, dependencies=['low', 'high'], max_window=11):
    # ((HIGH-LOW\u2013SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100
    sma = (data['high']-data['low']).ewm(adjust=False, alpha=float(2)/11, min_periods=0, ignore_na=False).mean()
    return ((data['high']-data['low']-sma)/sma) * 100

def alpha189(data, dependencies=['close'], max_window=12):
    # MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
    return abs(data['close']-data['close'].rolling(window=6,min_periods=6).mean()).rolling(window=6,min_periods=6).mean()

def alpha190(data, dependencies=['close'], max_window=40):
    # LOG((COUNT(RET>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)
    # *SUMIF((RET-(CLOSE/DELAY(CLOSE,19))^(1/20)-1)^2,20,RET<(CLOSE/DELAY(CLOSE,19))^(1/20)-1)
    # /(COUNT(RET<(CLOSE/DELAY(CLOSE,19))^(1/20)-1,20)
    # *SUMIF((RET-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,RET>(CLOSE/DELAY(CLOSE,19))^(1/20)-1)))
    ret = data['close'].pct_change(periods=1)
    ret_19 = (data['close']/data['close'].shift(19))**0.05-1.0
    part1 = (ret>ret_19).rolling(window=20, min_periods=20).sum()-1.0
    part2 = (np.minimum(ret-ret_19, 0.0) ** 2).rolling(window=20,min_periods=20).sum()
    part3 = (ret<ret_19).rolling(window=20, min_periods=20).sum()
    part4 = (np.maximum(ret-ret_19, 0.0) ** 2).rolling(window=20,min_periods=20).sum()
    return np.log(part1*part2/part3/part4)

def alpha191(data, dependencies=['volume', 'low', 'close', 'high'], max_window=25):
    # CORR(MEAN(VOLUME,20),LOW,5)+(HIGH+LOW)/2-CLOSE
    part1 = (data['volume'].rolling(window=20,min_periods=20).mean()).rolling(window=5,min_periods=5).corr(data['low'])
    return (part1 + data['high']*0.5+data['low']*0.5-data['close'])
