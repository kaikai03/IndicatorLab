
import sys
import os 
module_path = os.path.abspath(os.path.join('..')) 
if module_path not in sys.path: 
    sys.path.append(module_path)
    
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing
from sklearn import linear_model
import tools.Pretreat_Tools as pretreat
import tools.Sample_Tools as smpl

from base.JuUnits import excute_for_multidates
import QUANTAXIS as QA

# %load_ext autoreload
# %autoreload 2
# %aimport tools.Pretreat_Tools

def cal_ret_market(market_value,ret_excess_data):
    '''计算市场（平均）收益
        :param market_value：{pd.Series} --市值
        :param ret_excess_data：{pd.Series} --超额回报
        
        :return: {pd.Series}  -- 市值加权的市场平均收益
    '''
    ##  不取对数有时候有精度问题,权重不是精确1
    market_value_log = np.log(market_value)
    weight = market_value_log / market_value_log.sum()
    ret_market_f = (ret_excess_data * weight).sum()
    return ret_market_f
'''
    需要准备的基础数据
    stock_df = load_cache('all_train_qfq',cache_type=CACHE_TYPE.STOCK).sort_index()
    smpl.optimize_data_type(stock_df)
    # stock_df = pd.concat(list(map(lambda file:load_cache(file,cache_type=CACHE_TYPE.STOCK),['all_train_qfq','all_tail_qfq','all_older_qfq']))).sort_index()

    ## 日无风险回报
    ret_fs_data = pd.read_csv(module_path+'/data/static/china10yearbond.csv').set_index('date').sort_index()
    ret_fs_data = (ret_fs_data['high'].astype(np.float32)+ret_fs_data['low'].astype(np.float32))/2 * 0.01
    ret_fs_daily = ret_fs_data/252
'''

def prepare_data(stock_df,ret_fs):
    '''数据准备
        :param stock_df：{pd.DataFrame} --股票数据，包含close,volume,market_value，liquidity_market_value,industry
        :param ret_fs：{pd.Series} --无风险回报
        
        :return: {set in [ret_t,ret_t_excess,market_value_t,ret_excess_market_t]}
    '''

    ret_t = smpl.get_current_return(stock_df,'close')

    # 超额回报
    ret_t_excess = ret_t.groupby(pd.Grouper(level='date', freq='1M')).apply(
            lambda x:(x-ret_fs.get(x.index[0][0].strftime('%Y-%m'),default=ret_fs[-1])))

    # 市值
    market_value_t = stock_df['market_value']

    # 市场收益，日内全市场收益加权平均
    ret_excess_market_t = excute_for_multidates(ret_t_excess,
                                         lambda ret: cal_ret_market(market_value_t.loc[ret.index[0][0]],ret), 
                                         level=0)
    
    return ret_t, ret_t_excess, market_value_t, ret_excess_market_t

def camp_beta_alpha(ret_excess,ret_excess_market):
    '''beta和alpha因子
        :param ret_excess：{pd.Series} --超额回报
        :param ret_excess_market：{pd.Series} --市场收益，全市场收益加权平均
        
        :return: {pd.DataFrame}  -- 返回alpha，beta，以及残差
    '''
    # 5年daily单核执行约35分钟
    window=252
    half_life_window = 63
    half_life_ = list(map(lambda n:0.5**(n/half_life_window),range(1,window+1)))[::-1]
    half_life_weight = half_life_/np.sum(half_life_)

    
    res_tmp = []
    def reg(ret_t_ex):
        # print()
        # assert False,None
        X = ret_excess_market[ret_t_ex.index.get_level_values(0)]
        model = linear_model.LinearRegression(fit_intercept=True, n_jobs=1)
        res = model.fit(X.values.reshape(-1, 1),
                        ret_t_ex.values.reshape(-1, 1), 
                        sample_weight=half_life_weight)
        
        predict = model.predict([[X[-1]]])
        residual = ret_t_ex[-1] - float(predict)
        
        res_tmp.append({'date':ret_t_ex.index[-1][0],
                        'code':ret_t_ex.index[-1][1], 
                        'beta':float(res.coef_), 
                        'alpha':float(res.intercept_),
                        'residual':residual
                       })
        return 0
    
    ret_excess.dropna().groupby(level=1,group_keys=False).apply(
            lambda x:x.rolling(window).apply(reg))

    res_final = pd.DataFrame(res_tmp)
    res_final.set_index(['date', 'code'], inplace=True)
    res_final = res_final.sort_index()
    return res_final

def momentum(ret,ret_fs):
    '''动量因子
        :param ret：{pd.Series} --回报率
        :param ret_fs：{pd.Series} --无风险回报率
    '''
    ret_excess = ret.groupby(pd.Grouper(level='date', freq='1M')).apply(
            lambda x:np.log(1+x)-np.log(1+ret_fs.get(x.index[0][0].strftime('%Y-%m'),default=ret_fs[-1])))

    def calc_(data,window=252,half_life_window=126):
        if len(data) < 253:
            return None
        ewma = data.rolling(window).apply(
                        lambda xx:(xx.ewm(adjust=False,halflife=126).mean()[-1]))
        return ewma.rolling(11).mean().shift(11)

    mom = excute_for_multidates(ret_excess.dropna(), lambda x:calc_(x), level='code')
    mom.name = 'momentum'
    return mom

def sizelg(stock_data):
    '''市值因子
        :param stock_data：{pd.DataFrame} --需要包含market_value
    '''
    mv = np.log(stock_data['market_value'])
    mv.name = 'sizelg'
    return mv

def bp(stock_data):
    '''Book-to-Price
        :param stock_data：{pd.DataFrame} --需要包含close
    '''
    data = smpl.add_report_inds(stock_data[['close']],'netAssetsPerShare')
    bp = data['close']/data['netAssetsPerShare']
    bp.name = 'bp'
    return bp


def earnings_yield(ret,market_value,stock_industry):
    '''Earnings Yield 收益因子   
        :param ret：{pd.Series} --回报率
        :param market_value：{pd.Series} --市值
        :param stock_industry：{pd.Series} --行业

        --EARNYILD = 0.68*EPIBS + 0.11*ETOP + 0.21*CETOP
        --EPIBS ：分析师预测的 EP （ earnings to price ）。
        --ETOP ： ttm-ep ，最近 12 个月的总盈利除以当前总市值。
        --CETOP ：最近 12 个月的运营现金流处于当前总市值。
        !!!!注意：需要全局回归(行业期望)，禁止使用分布计算
    '''
    mv = market_value
    codes = mv.index.get_level_values(1).unique().tolist()
    date_ = mv.index.get_level_values(0)
    date_start = str(int(date_.min().strftime("%Y"))-1)
    date_end = date_.max().strftime("%Y")

    # # 利润总额  经营活动产生的现金流量净额  
    report_df = QA.QA_fetch_financial_report_adv(codes, 
                                                 date_start, date_end,ltype='EN'
                                                ).data[['totalProfit',
                                                        'netCashFlowsFromOperatingActivities']]


    # 年报转累进转当季
    report_df = excute_for_multidates(report_df,
                                      lambda stock:stock.groupby(pd.Grouper(level='report_date', freq='1Y')).apply(
                                      lambda x:x.diff(1).fillna(x)),level='code')

    # 四季（年）滚动加总，“最近12个月”
    report_df = excute_for_multidates(report_df,lambda x:x.rolling(4).sum(),level='code')

    data_ = excute_for_multidates(pd.concat([mv,report_df], axis=1),
                                  lambda x:x.fillna(method='ffill'),level='code'
                                 ).loc[mv.index].sort_index()
    
    ETOP = data_['totalProfit']/data_['market_value']
    CETOP = data_['netCashFlowsFromOperatingActivities']/data_['market_value']


    # # EPIBS 分析师的期望暂时用季度收益斜率+行业季度收益斜率来代表。
    def ret_cum_reg(ret,window=63):
        def reg(window_slice):
            ## ！X设置与同一量级
            m = linear_model.LinearRegression(fit_intercept=True, n_jobs=1)
            res = m.fit(np.arange(0.01,0.01*window+0.01,0.01).reshape(-1, 1), 
                        window_slice.values.reshape(-1, 1)
                       )
            return float(res.coef_)

        k = np.log(1+ret).rolling(window).apply(lambda x:reg(x))
        return k

    ret_expect = excute_for_multidates(ret, lambda x:ret_cum_reg(x),level='code').sort_index()

    ret_industry = pd.concat([ret,stock_industry], axis=1).sort_index()
    # 日内行业平均
    ret_industry_meam = ret_industry.reset_index().set_index(['date','industry']).groupby(level=[0,1]).mean()
    ret_industry_expect = excute_for_multidates(ret_industry_meam, lambda x:ret_cum_reg(x),level='industry')

    EPIBS = ret_expect + ret_industry_expect.loc[list(zip(ret_industry.index.get_level_values(0),ret_industry['industry']))]['ret'].values

    # # # 测试
    # with pd.option_context("display.max_rows", None, "display.max_columns", None):
    #     x = pd.DataFrame(ret_industry_expect.loc[list(zip(ret_industry.index.get_level_values(0),ret_industry['industry']))].values,index=ret_industry.index)
    #     display(pd.concat([ret_industry,ret_expect,x],axis=1))

    EARNYILD = 0.68*EPIBS + 0.11*ETOP + 0.21*CETOP
    EARNYILD.name = 'earnings_yield'
    return EARNYILD


def liquidity(data_df):
    '''流动性因子   
        :param data_df：{pd.DataFrame} --需要包含流动市值liquidity_market_value、close和market_value
        --LIQUIDTY = 0.35*STOM + 0.35*STOQ + 0.30*STOA 
        --STOM: 月均换手率：ST(1)
        --STOQ ：三个月的平均月换手率：ST(3)
        --STOA ：十二个月的平均月换手率：ST(12)
        !!!!注意：需要全局回归，禁止使用分布计算
    '''
    liquidity_capital = data_df['liquidity_market_value']/data_df['close']
    turn_over = data_df['volume']*100 / liquidity_capital
    turn_over_month_sum = excute_for_multidates(turn_over, lambda x:x.rolling(21).sum(),level='code')
    STOM = np.log(turn_over_month_sum)
    STOQ = np.log(excute_for_multidates(turn_over_month_sum, lambda x:x.rolling(21*3).mean(),level='code'))
    STOA = np.log(excute_for_multidates(turn_over_month_sum, lambda x:x.rolling(21*3*4).mean(),level='code'))
    LIQUIDTY = 0.35*STOM + 0.35*STOQ + 0.30*STOA 
    
    liqiodty_no_nan = LIQUIDTY.dropna()
    size = np.log(data_df['market_value'].loc[liqiodty_no_nan.index])
    
    def reg(y):
        Y = y.dropna().values.reshape(-1, 1)
        X = size.loc[y.index].values.reshape(-1, 1)
        model = linear_model.LinearRegression(fit_intercept=True)    
        resualt = model.fit(X, Y)
        predict = resualt.predict(X)
        residual = y - np.squeeze(predict)
        return residual
    
    liquidity_residual = excute_for_multidates(liqiodty_no_nan, lambda y:reg(y),level='date')
    LIQUIDTY[liquidity_residual.index] = liquidity_residual.values
    LIQUIDTY.name = 'liquidity'
    return LIQUIDTY

def resvol(ret, ret_fs, ret_excess, size_log, beta, beta_residual):
    '''Residual Volatility 波动因子
       可以认为 beta 之外的剩余风险
        :param ret：{pd.Series} --回报率
        :param ret_fs：{pd.Series} --无风险回报
        :param ret_excess：{pd.Series} --超额回报
        :param beta：{pd.Series} --beta，需要先计算camp_beta_alpha
        :param beta_residual：{pd.Series} --beta残差，需要先计算camp_beta_alpha
        --RESVOL = 0.74*DASTD + 0.16 *CMRA + 0.10*HSIGMA
        --DASTD：Daily std：日标准差。超额收益率序列半衰加权标准差，T=252，半衰期为42天。
        --CMRA：Cumulative range：累积收益范围。表示过去12个月的波动率幅度。每21天计算一个Z(T)。
        --HSIGMA：Hist sigma：历史sigma，在计算Beta所进行的时间序列回归中，取回归残差收益率的波动率。
        --最后RESVOL对beta做正交化
        !!!!注意：需要全局回归，禁止使用分布计算
    '''

    window=252
    half_life_window = 42
    half_life_ = list(map(lambda n:0.5**(n/half_life_window),range(1,window+1)))[::-1]
    half_life_weight = half_life_/np.sum(half_life_)

    DASTD = excute_for_multidates(ret_excess, 
                                  lambda stock:stock.rolling(window).apply(
                                      lambda x:((x-x.mean())**2).dot(half_life_weight)**(0.5)
                                  ) ,level='code')

    ret_excess_log = excute_for_multidates(ret,
                                           lambda x:np.log(1+x)-np.log(1+ret_fs.get(x.index[0][0].strftime('%Y-%m'),default=ret_fs[-1])),
                                           level='code')
    
    Z = excute_for_multidates(ret_excess_log,
                              lambda x:x.rolling(21).sum(),
                              level='code')
    
    CMRA = excute_for_multidates(Z,
                              lambda x:x.rolling(21).apply(lambda x:x.max()-x.min()),
                              level='code')
    
    HSIGMA = excute_for_multidates(beta_residual,
                                  lambda x:x.rolling(252).std(),
                                  level='code')
    
    RESVOL = 0.74*DASTD + 0.16 *CMRA + 0.10*HSIGMA
    
    model = linear_model.LinearRegression(fit_intercept=True)
    
    
    resvol_no_nan = RESVOL.dropna()
    size_ = size_log.loc[resvol_no_nan.index]
    beta_ = beta.loc[resvol_no_nan.index]    
    
    def reg(y_):
        Y = y_.values.reshape(-1, 1)
        x1 = size_.loc[y_.index].values.reshape(-1, 1)
        x2 = beta_.loc[y_.index].values.reshape(-1, 1)
        X = np.concatenate((x1, x2),axis=1)
        
        model = linear_model.LinearRegression(fit_intercept=True)    
        resualt = model.fit(X, Y)
        predict = resualt.predict(X)
        residual = y_ - np.squeeze(predict)
        return residual
    
    resvol_residual = excute_for_multidates(resvol_no_nan, lambda y:reg(y),level='date')
    RESVOL[resvol_residual.index] = resvol_residual.values
    RESVOL.name = 'resvol'
    
    return RESVOL


def sizenl(size_lg):
    '''Non-Linear Size 非线性市值因子
    :param ret：{pd.Series} --回报率
    :param ret_fs：{pd.Series} --无风险回报
    --市值对数LNCAP的立方和LNCAP做线性回归后的残差，再经过缩尾处理(winsorized)和标准化处理(standardized)。
    --可代表"中市值因子"，相当于x^3用一条均线穿过去分为上下部分（残差正负，负的部分在中间）
    !!!!注意：需要全局回归，禁止使用分布计算
    '''
    size_lg_ = size_lg.copy()
    size_nona= size_lg_.dropna()
    
    def standardize(x):
        x_3 = x**3
        Y = x_3.values.reshape(-1, 1)
        X = x.values.reshape(-1, 1)

        model = linear_model.LinearRegression(fit_intercept=True)    
        resualt = model.fit(X, Y)
        predict = resualt.predict(X)
        residual = x_3 - np.squeeze(predict)
        sizenl = pretreat.winsorize_by_mad(residual, n=3, drop=False)
        sizenl = pretreat.standardize(residual, multi_code=False)
        return sizenl

    SIZENL_standarded = excute_for_multidates(size_nona, lambda x:standardize(x),level='date')
    size_lg_[SIZENL_standarded.index] = SIZENL_standarded
    
    size_lg_.name = 'sizenl'
    return size_lg_
