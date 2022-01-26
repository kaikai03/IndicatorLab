
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as st
import statsmodels.api as sm
import QUANTAXIS as QA

from sklearn import linear_model

import numba as nb

def get_LR_params_fast(x_array,y_array):
    """快速最小二乘：
        :param x_array：{list or np.array}
        :param y_array：{list or np.array}
        :return：{list} -- [-1]为截距，其他为系数
    """
    if not isinstance(x_array, np.ndarray):
        x= np.array(x_array)
    else:
        x=x_array
        
    if len(x.shape) < 2:
        x=x.reshape(-1,1)
    x = np.hstack([x,np.ones([x.shape[0],1])])
    params = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y_array)
    return params

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
    
    df = pd.DataFrame({'factor_standardized':factor_standardized.loc[common_index], 'ret_f':ret_forward})
    df = df[(~pd.isnull(df['factor_standardized'])) & (~pd.isnull(df['ret_f']))]
    

    # 计算相关系数
    for dt in df.index.get_level_values(0).unique():
        if len(df.loc[dt]) < 5:
            #'参与计算标的小于5只时，跳过'
            continue
        ic = df['factor_standardized'][dt].rank().corr(df['ret_f'][dt].rank(),method='pearson')
        ic_data[dt] = ic
        
    return ic_data

def get_ic_desc(ic_data):
    """因子信息系数的描述
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
    """因子胜率
       :param ic_data_df:{pd.DataFrame，Index[date,]} --rankIC值, 
       :return: {pd.DataFrame}
    """
    df_ = ic_data_df.dropna()
    ic_mean_sign = np.sign(df_.mean())
    return (np.sign(df_) == ic_mean_sign).sum()/df_.count()

def climbing(data:np.ndarray):
    '''上升
       :param data: ;
       :return：{tuple} -- tuple(最大上升,上升率,最大上升周期,底下标,顶下标)
    '''
    top_index = np.argmax(data-np.minimum.accumulate(data))
    if top_index == 0:
        return 0, 0, 0, 0, 0
    bottom_index = np.argmin(data[:top_index])
    
    duration = bottom_index - top_index
    up = data[bottom_index] - data[top_index]
    up_rate = np.round(up / data[bottom_index], 2)
    return up, up_rate, duration, bottom_index, top_index


def retracing(data:np.ndarray):
    '''回撤
       :param data: ;
       :return：{tuple} -- tuple(最大回撤,回撤率,最大回撤周期,顶下标,底下标)
    '''
    bottom_index = np.argmax(np.maximum.accumulate(data)-data)
    if bottom_index == 0:
        return 0, 0, 0, 0, 0
    top_index = np.argmax(data[:bottom_index])
    
    retrace_duration = bottom_index - top_index
    retrace = data[top_index] - data[bottom_index]
    retrace_rate = np.round(retrace / data[top_index], 2)
    return retrace, retrace_rate, retrace_duration, top_index, bottom_index

def cross_sign(direct_signals:np.ndarray):
    """根据数据方向序列，取得驻点
    """
    ## 把0 bfill掉，避免0的时候误认为拐点
    mask0 = direct_signals==0
    idx0 = np.where(~mask0,np.arange(mask0.shape[0]),0)
    np.maximum.accumulate(idx0, out=idx0)
    signal = direct_signals[idx0]
    
    dif_shift_neg1 = np.append(signal[1:],0)
    cross_sign = np.where(signal==dif_shift_neg1,0,np.where(signal<dif_shift_neg1,1,-1))
    cross_sign[0]=0
    cross_sign[-1]=0
    return cross_sign

def get_direct_sign(arr:np.ndarray):
    """取得序列变化方向
    """
    direct_sign = np.append(0,np.sign(np.diff(arr)))
    return direct_sign

def get_cross_sign(direct_signals:np.ndarray):
    """根据数据方向序列，取得驻点
    """
    ## 把0 bfill掉，避免0的时候误认为拐点
    mask0 = direct_signals==0
    idx0 = np.where(~mask0,np.arange(mask0.shape[0]),0)
    np.maximum.accumulate(idx0, out=idx0)
    signal = direct_signals[idx0]
    
    dif_shift_neg1 = np.append(signal[1:],0)
    cross_sign = np.where(signal==dif_shift_neg1,0,np.where(signal<dif_shift_neg1,1,-1))
    cross_sign[0]=0
    cross_sign[-1]=0
    return cross_sign


def get_longest_recover_duration(data:np.ndarray, cross_sign_arr:np.ndarray):
    '''计算恢复原点位所用的周期
       :param cross_sign_arr: --拐点信号序列，1上折，-1下折，0保持;
       :return：{tuple} -- tuple(最大周期,起点下标,终点下标)
    '''
    cross_down_idx = np.argwhere( cross_sign_arr==-1 )  # 下折点统计
    max_recover_duration = 0 #默认为无回复周期
    max_recover_start = None
    max_recover_end = None
    # 查找整个窗口内的最大回复周期，既从下折开始，恢复到原始价格所用的时间窗口
    if len(cross_down_idx) > 0:
        for cd_idx in cross_down_idx.T[0]:
            #遍历下折点
            cross_value = data[cd_idx]
            #从下折点的下一个点开始找回复点
            recovers = np.argwhere(data[cd_idx+1:]>=cross_value)

            if len(recovers)>0:
                # 第一个恢复点的下标，相当于本次下折的最小回复周期
                # 由于下标起点为0，所以实际周期+1,这个周期不包括起点本身
                recover_duration = recovers.T[0][0]+1 
                # 更新最大周期
                if recover_duration > max_recover_duration:
                    max_recover_duration = recover_duration
                    max_recover_start = cd_idx #本身为起点
                    max_recover_end = cd_idx+recover_duration
    return max_recover_duration, max_recover_start, max_recover_end


def get_longest_rollercoaster_duration(data:np.ndarray, cross_sign_arr:np.ndarray):
    '''计算最大过山车周期
       :param cross_sign_arr: --拐点信号序列，1上折，-1下折，0保持;
       :return：{tuple} -- tuple(最大周期,起点下标,终点下标)
    '''
    cross_up_idx = np.argwhere( cross_sign_arr==1 )  # 上折点统计
    max_rollercoaster_duration = 0 #默认为无回复周期
    max_rollercoaster_start = None
    max_rollercoaster_end = None
    # 查找整个窗口内的最大回复周期，既从上折开始，恢复到原始价格所用的时间窗口
    if len(cross_up_idx) > 0:
        #遍历上折点
        for cu_idx in cross_up_idx.T[0]:
            cross_value = data[cu_idx]
            #从上折点的下一个点开始找回归点
            regorge = np.argwhere(data[cu_idx+1:]<=cross_value)
            if len(regorge)>0:
                # 第一个恢复点的下标，相当于本次下折的最小回复周期
                # 由于下标起点为0，所以实际周期+1,这个周期不包括起点本身
                rollercoaster_duration = regorge.T[0][0]+1 
                # 更新最大周期
                if rollercoaster_duration > max_rollercoaster_duration:
                    max_rollercoaster_duration = rollercoaster_duration
                    max_rollercoaster_start = cu_idx #本身为起点
                    max_rollercoaster_end = cu_idx + rollercoaster_duration
    return max_rollercoaster_duration, max_rollercoaster_start,max_rollercoaster_end


@nb.jit(nopython=True)
def ma_fast(data:np.array, window:int):
    '''
        快速MA,数据中禁止存在nan
        不考虑jit的话，可以使用下面的strided_rolling
        例：ma_fast(x[~np.isnan(x)],window)
    '''
    assert np.isnan(data).any()==False ,'can not contain nan'
    sum_ = 0
    result = np.full(len(data),np.nan)
    
    for i in range(0, len(data)):
        sum_ += data[i]
        if i > window-1:
            sum_ -= data[i-window]
        if i >= window-1:
            result[i] = sum_ / window
    return result

@nb.jit(nopython=True)
def inv_num_fast(series):
    '''
        计算逆序数个数
    '''
    count = 0
    for i in range(len(series)):
        x = series[i]
        for j in range(i):
            if x < series[j]:
                count+=1
    return count
    
@nb.jit(nopython=True)
def ma_power_np_base(data, ma_param=np.array(range(5, 30))):
    '''
        MA均线多头排列能量强度
        :param data:{nb.typed.List} --价格或指标序列;
        :param ma_param:{np.array} --移动平均参数;
        :return：{list} [0-1] 
    '''
    assert len(ma_param) > 1, 'size of ma_param must > 1'
    ma_np = np.empty((len(data), len(ma_param)))
    ma_count = 0

    for r in ma_param:
        sum = 0
        for i in range(0, len(data)):
            sum += data[i]
            if i > r-1:
                sum -= data[i-r]
            if i >= r-1:
                ma_np[i, ma_count]=sum/r
        ma_count += 1

    ma_max = max(ma_param)
    param_len = len(ma_param)
    param_count = (param_len * (param_len - 1)) / 2
    
    num = np.zeros(len(data))
    ratio = np.zeros(len(data))
#     with np.errstate(invalid='ignore', divide='ignore'):
    for i in range(ma_max,len(data)):
        # 排在数组越靠前，周期就小，
        # 在多头趋势中，理论上应该最快反应，也就是值越大。所以逆序能代表其强度
        num[i] = inv_num_fast(ma_np[i, :]) 
        ratio[i] = num[i] / param_count

    return ratio

def ma_power_np(data_series, ma_param=range(5, 30)):
    '''
        MA均线多头排列能量强度
        :param data_series:{pd.Series} --价格或指标序列;
        :param ma_param:{list} --移动平均参数;
        :return：{pd.Series} [0-1] 
    '''
    param = np.array(ma_param)
    res = ma_power_np_base(nb.typed.List(data_series.to_list()), param)
    return pd.Series(res,index=data_series.index)


@nb.jit(nopython=True)
def peak_detection_z(y, lag = 5, std_times = 3, convolute_coef = 0.5):
    """
        Robust peak detection algorithm (using z-scores)
        自带鲁棒性极值点识别，利用方差和ZSCORE进行时间序列极值检测。算法源自：
        https://stackoverflow.com/questions/22583391/
        本实现使用Numba JIT优化，比原版（上面）大约快了500倍。
        :param y:{np.array | nb.typed.List} --数据;
        :param std_times:{float} --偏差几倍于标准差;
        :param convolute_coef:{float}[0-1] --越过标准差边界时，
                下一论被纳入计算的当前y，与y_t-1加权平均;既系数越小，状态的持续性越强。
        :return：{list[3,]}  --list[0]:[-1,0,1]点状态；list[1]:均值序列；list[2]:标准差序列；
    """
    ret_signals = np.zeros((3, len(y),))
    idx_signals = 0
    idx_avgFilter = 1
    idx_stdFilter = 2

    filteredY = np.copy(y)
    ret_signals[idx_avgFilter, lag - 1] = np.mean(y[0:lag])
    ret_signals[idx_stdFilter, lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - ret_signals[idx_avgFilter, i - 1]) > std_times * ret_signals[idx_stdFilter, i - 1]:
            if y[i] > ret_signals[idx_avgFilter, i - 1]:
                ret_signals[idx_signals, i] = 1
            else:
                ret_signals[idx_signals, i] = -1

            filteredY[i] = convolute_coef * y[i] + (1 - convolute_coef) * filteredY[i - 1]
            ret_signals[idx_avgFilter, i] = np.mean(filteredY[(i - lag):i])
            ret_signals[idx_stdFilter, i] = np.std(filteredY[(i - lag):i])
        else:
            ret_signals[idx_signals, i] = 0
            filteredY[i] = y[i]
            ret_signals[idx_avgFilter, i] = np.mean(filteredY[(i - lag):i])
            ret_signals[idx_stdFilter, i] = np.std(filteredY[(i - lag):i])

    return ret_signals

def peak_detection(data_series, lag = 5, std_times = 3, convolute_coef = 0.5):
    """
        Robust peak detection algorithm (using z-scores)
        :param y:{list|np.array|pd.Series} --数据;
        :param std_times:{float} --偏差几倍于标准差;
        :param convolute_coef:{float}[0-1] --越过标准差边界时，
                下一论被纳入计算的当前y，与y_t-1加权平均;既系数越小，状态的持续性越强。
        :return：{list[3,]}  --list[0]:[-1,0,1]点状态；list[1]:均值序列；list[2]:标准差序列；
    """
    ret_signals = peak_detection_z(np.array(data_series), lag = lag, std_times = std_times, convolute_coef = convolute_coef)
    if isinstance(data_series, pd.Series):
        return pd.Series(ret_signals[0],index=data_series.index)
    else:
        return pd.Series(ret_signals[0])


def strided_rolling(a, L, S):  
    '''
    Pandas rolling for numpy
    # Window len = L, Stride len/stepsize = S
    '''
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S * n,n))

def rolling_poly(data_arr:np.ndarray, window:int=252, ploy_vars_count=9) -> np.ndarray:
    '''
    一元九项式滚动分解拟合
    '''
    x_index = range(window)
    def last_poly(sx):
        p = np.polynomial.Chebyshev.fit(x_index, sx, ploy_vars_count)
        return p(x_index)[-1]

    if (len(data_arr) > window):
        x = strided_rolling(data_arr, window, 1)
        return np.r_[np.full(window - 1, np.nan), 
                     np.array(list(map(last_poly, x)))]
    else:
        x_index = range(len(data_arr))
        p = np.polynomial.Chebyshev.fit(x_index, data_arr, ploy_vars_count)
        y_fit_n = p(x_index)
        return y_fit_n

def poly_effectiveness(data_arr:np.ndarray, window:int=252, ploy_vars_count:int=5, get_fitted:bool=False):
    '''
        多项式滚动拟合相关性 
        --经验值：0.5左右及以下为随机序列。真实价格序列>0.8，另外且方差下降，也说明了波动在增大
        :param data_arr:{np.ndarray} 
        :param window:{int}  --回望周期
        :param ploy_vars_count:{int}  --多项式项数,5时，各种随机生成算法的结果约0.5
        :param get_fitted:{bool} --True时返回拟合数据。 
        :return：{float[0-1]|tuple(folat,np.ndarray)} 
    '''
    fitted = rolling_poly(data_arr, window, ploy_vars_count)
    cor = np.corrcoef(data_arr[window-1:],fitted[window-1:])
    if get_fitted:
        return cor[0][1],fitted
    return cor[0][1]


def RPS_rank_backward(stock_df, deal_cloumn='close', ret_stride=5):
    """
        来源：RPS英文全称Relative Price Strength Rating，即股价相对强度，
        该指标是欧奈尔CANSLIM选股法则中的趋势分析，具有很强的实战指导意义。
        RPS指标是指在一段时间内，个股涨幅在全部股票涨幅排名中的位次值。
        
        这个函数计算每个交易日所有收益排行的《反序》，既收益越大，排名的数值越大。
        :param stock_df:{np.DataFrame} 
        :param deal_cloumn:{str}  --所要处理的字段
        :param ret_stride:{int}  --回报周期
        :return：{np.DataFrame} 
    """
    rets = smpl.get_current_return(stock_df,deal_cloumn,ret_stride)
    rets.name = deal_cloumn+'_pct'
    
    def get_RPS(series):
        """计算RPS(时间截面)
        """
        ret_rank = pd.DataFrame(series)
        ret_rank['rank'] = series.rank(ascending=True, pct=True)
        return ret_rank
    
    ret_indices = pd.DataFrame(rets)
#     ret_indices[deal_cloumn] = df[deal_cloumn]
    ret_indices['rank'] = np.nan
    ret_indices.loc[:, ['rank']] = rets.groupby(level=[0]).apply(lambda x:get_RPS(x))
    return ret_indices


@nb.jit(nopython=True)
def LIS(X:np.array) -> tuple:
    """计算最长递增子序列
       :return: {tuple(list(序列值), list(序列idx))}
    """
    N = len(X)
    P = [0] * N
    M = [0] * (N + 1)
    L = 0
    for i in range(N):
        lo = 1
        hi = L
        while lo <= hi:
            mid = (lo + hi) // 2
            if (X[M[mid]] < X[i]):
                lo = mid + 1
            else:
                hi = mid - 1

        newL = lo
        P[i] = M[newL - 1]
        M[newL] = i

        if (newL > L):
            L = newL

    S = []
    pos = []
    k = M[L]
    for i in range(L - 1, -1, -1):
        S.append(X[k])
        pos.append(k)
        k = P[k]
    return S[::-1], pos[::-1]


@nb.jit(nopython=True)
def LDS(X:np.array) -> tuple:
    """计算最长递减子序列 Longest decreasing subsequence
       :return: {tuple(list(序列值), list(序列idx))}
    """
    N = len(X)
    P = [0] * N
    M = [0] * (N + 1)
    L = 0
    for i in range(N):
        lo = 1
        hi = L
        while lo <= hi:
            mid = (lo + hi) // 2
            if (X[M[mid]] > X[i]):
                lo = mid + 1
            else:
                hi = mid - 1

        newL = lo
        P[i] = M[newL - 1]
        M[newL] = i

        if (newL > L):
            L = newL

    S = []
    pos = []
    k = M[L]
    for i in range(L - 1, -1, -1):
        S.append(X[k])
        pos.append(k)
        k = P[k]
    return S[::-1], pos[::-1]


@nb.jit(nopython=True)
def timeline_event_integral(signals:np.array) -> np.array:
    """
        事件发生的连续性计数
        计算时域金叉/死叉信号的累积卷积和(死叉(1-->0)清零)，
        经测试for实现最快，比reduce快
        :param signals:{np.array} [0|1]  
                -- jx = timeline_integral(np.where(data.diff(1) > 0, 1,0))
                   sx = timeline_integral(np.where(data.diff(1) < 0, 1,0))
        :return: {np.array}
    """
    t = np.zeros(len(signals)).astype(np.int32)
    for i, s in enumerate(signals):
        t[i] = s * (t[i - 1] + s)
    return t


@nb.jit(nopython=True)
def timeline_event_duration(signals:np.array) -> np.array:
    """
        事件发生的《间隔》,既：无事发生的连续性计数
        计算时域金叉/死叉信号的累积卷积和(死叉(1-->0)不清零，金叉(0-->1)清零)
        经测试for最快，比reduce快(无jit，jit的话for就更快了)
        :param signals:{np.array[0|1] } 
        :return: {np.array}
    """
    t = np.zeros(len(signals)).astype(np.int32)
    for i, s in enumerate(signals):
        t[i] = (t[i - 1] + 1) if (s != 1) else 0
    return t

def timeline_event_continuity(directions):
    """
    计算事件发生的连续性计数，含方向
    !!!!注意，0不计入事件。
        不要直接简单传递diff差值，无法直接处理差分为0的情况。
        传递sign更合理
    
    :param tm:{np.array[-1|0|1]} 
    :return: {np.array}
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        trend_jx = timeline_event_integral(np.where(directions > 0, 1, 0))
        trend_sx = np.sign(directions) * timeline_event_integral(np.where(directions < 0, 1, 0))
    return np.array(trend_jx + trend_sx).astype(np.int32)

def feature_JXSX_timeline(serise):
    """
    技术指标特征的金叉死叉时序
    !!!!注意，不支持输入0，既无事件的间隔
        0会被认为是事件翻转
    :param tm:{np.array[-1|1]} 
    :return: {np.array}
    实际使用上方的timeline_event_continuity更好，支持0跳过。
    """
    features_jx_before = timeline_event_duration(np.where(serise > serise.shift(1), 1, 0))
    features_sx_before = timeline_event_duration(np.where(serise < serise.shift(1), 1, 0)) 
    return timeline_event_continuity(np.where((features_jx_before < features_sx_before), 1, -1))

