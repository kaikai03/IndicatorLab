
import sys
import os 
module_path = os.path.abspath(os.path.join('..')) 
if module_path not in sys.path: 
    sys.path.append(module_path)
    
import numpy as np
import pandas as pd

import Analysis_Funs as af

import tools.Sample_Tools as smpl
import tools.Pretreat_Tools as pretreat

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import ind.Ind_Model_Base as Ind_Model_Base
from base.Constants import PLOT_TITLE

from base.JuUnits import parallal_task,task_chunk_split
import dill

def FactorTest_deal(codes,self_obj):
    import Analysis_Funs as af
    import tools.Sample_Tools as smpl
    import tools.Pretreat_Tools as pretreat
    import dill
    import pandas as pd

    self_ = dill.loads(self_obj)

    data = smpl.get_data(codes, start=self_.start, end=self_.end, gap=self_.gap)

    df = smpl.resample_stockdata_low(data.data,freq=self_.freq)
    ret_forward_re = smpl.get_forward_return(df,'close')

    # 为保证ind_Model_Class信息的完整性，使用重采样前的数据来生成指标，后面会另外再重采样。
    ind_obj = self_.ind_Model_Class(data.data) 
    ind_obj.fit()
    ind = pd.DataFrame(ind_obj.ind_df[self_.main_field]) 
    ind.dropna(axis=0,inplace=True)
    if self_.neutralize.get('enable',False):
        ind_close = pd.concat([ind, data.close], axis=1)#为了给复权用
        ind_close.dropna(axis=0,inplace=True)
        ind_added = smpl.add_marketvalue_industry(ind_close, static_mv=self_.neutralize.get('static_mv',False))
        return (ind_added,ret_forward_re)

    return (ind,ret_forward_re)


class FactorTest():
    def __init__(self,ind_Model_Class, sample='上证50',freq="m",start=None,end=None,gap=2500,only_main=True,neutralize={'enable':False,'static_mv':False},target_field=None):
        assert ((not start is None) or (not end is None)), 'start 和 end 必须有一个'
        assert isinstance(ind_Model_Class,type(Ind_Model_Base.Ind_Model)),"ind_Model_Class必须是Ind_Model的子类"
        assert freq !='w' ,"freq 禁止直接写w，自动resample week经常会选在周末而导致计算ic没有交集"
        
        self.ind_Model_Class = ind_Model_Class
        if target_field is None:
            self.main_field = ind_Model_Class.optimum_param['main']
        else:
            self.main_field = target_field
        self.sample = sample
        self.freq = freq
        self.start = start
        self.end = end
        self.gap = gap
        self.only_main = only_main
        self.neutralize=neutralize
        self.rank_ic = None
        self.res = None
        self.ind_ret_df = None
        self.ind_binned = None
        

    def process(self):
        data = smpl.get_sample_by_zs(name=self.sample, start=self.start, end=self.end, gap=self.gap, only_main=self.only_main)
        
        df = smpl.resample_stockdata_low(data.data,freq=self.freq)
        # 后续的重采样依赖于ret_forward，否则不同周期下，resample会出现日期不一致的情况。
        ret_forward = smpl.get_forward_return(df,'close')
        
        # 为保证ind_Model_Class信息的完整性，使用重采样前的数据来生成指标，后面会另外再重采样。
        ind_obj = self.ind_Model_Class(data.data) 
        ind_obj.fit()
        ind = pd.DataFrame(ind_obj.ind_df[self.main_field])
        ind.dropna(axis=0,inplace=True)
        
        
        if self.neutralize.get('enable',False):
            ind_close = pd.concat([ind, data.close], axis=1)#为了给复权用
            ind_close.dropna(axis=0,inplace=True)
            ind_added = smpl.add_marketvalue_industry(ind_close, static_mv=self.neutralize.get('static_mv',False))
#             self.indx1 = ind_added
            x = ind_added[['totalCapital','industry']].sort_index()
            # x = ind_added[['liquidity_totalCapital','industry']]
            y = ind_added.iloc[:,0].sort_index()
            ind = pretreat.neutralize(y, x, categorical=['industry'], logarithmetics=['totalCapital'])
            
             # 取消因子标准化，很多时候标准化后的rank_ic的结果，与分箱测试观测结果不符
#             factor_standardized = pretreat.standardize(ind, multi_code=True)
            self.rank_ic = af.get_rank_ic(ind, ret_forward)
        else:
            # neutralize 最后得到的ind是series，而原来的是dataframe
            # get_rank_ic 内部会做交集，这外面就不必resample了
#             factor_standardized = pretreat.standardize(ind, multi_code=True)[self.main_field]
            self.rank_ic = af.get_rank_ic(ind[self.main_field], ret_forward)
        
#         self.a = ind
#         self.b = ret_forward
        
#         self.rank_ic = af.get_rank_ic(factor_standardized, pretreat.standardize(ret_forward, multi_code=True))
        
        self.res = pd.DataFrame([af.get_ic_desc(self.rank_ic)], columns=['rankIC','rankIC_std','rankIC_T','rankIC_P'])
        self.res['ICIR']=round(af.get_ic_ir(self.rank_ic),6)
        self.res['winning']=round(af.get_winning_rate(self.rank_ic),6)
        
        
        common_index = ind.index.get_level_values(0).unique().intersection(ret_forward.index.get_level_values(0).unique())
        ind_resample = ind.loc[common_index]
        self.ind_ret_df = pd.concat([ind_resample, ret_forward], axis=1)
        self.ind_ret_df.dropna(axis=0,inplace=True)
        # 分箱
        self.ind_binned = self.ind_ret_df.groupby(level=0, group_keys=False).apply(lambda x: pretreat.binning(x, deal_column=self.main_field,box_count=10, inplace=True))
        
        
    def process_multi(self, worker=4):
        codes = smpl.get_codes_by_zs(name=self.sample, only_main=self.only_main)
        task = task_chunk_split(codes, worker)
        
        results = parallal_task(worker, FactorTest_deal, task, self_obj=dill.dumps(self))
        
        res_T = np.array(results,dtype=object).T.tolist()
        ind = pd.concat(res_T[0])
        ret_forward_re = pd.concat(res_T[1])
#         self.indx1 = ind 
        
        if self.neutralize.get('enable',False):
            x = ind[['totalCapital','industry']].sort_index()
            # x = ind[['liquidity_totalCapital','industry']]
            y = ind.iloc[:,0].sort_index()
            ind = pretreat.neutralize(y, x, categorical=['industry'], logarithmetics=['totalCapital'])
        
#         factor_standardized = pretreat.standardize(ind, multi_code=True)
            self.rank_ic = af.get_rank_ic(ind, ret_forward_re)
        else:
            self.rank_ic = af.get_rank_ic(ind[self.main_field], ret_forward_re)

#         self.rank_ic = af.get_rank_ic(factor_standardized, ret_forward_re)

        

        self.res = pd.DataFrame([af.get_ic_desc(self.rank_ic)], columns=['rankIC','rankIC_std','rankIC_T','rankIC_P'])
        self.res['ICIR']=round(af.get_ic_ir(self.rank_ic),6)
        self.res['winning']=round(af.get_winning_rate(self.rank_ic),6)
        
        common_index = ind.index.get_level_values(0).unique().intersection(ret_forward_re.index.get_level_values(0).unique())
        ind_resample = ind.loc[common_index]
        self.ind_ret_df = pd.concat([ind_resample, ret_forward_re], axis=1)
        self.ind_ret_df.dropna(axis=0,inplace=True)
        # 分箱
        self.ind_binned = self.ind_ret_df.groupby(level=0, group_keys=False).apply(lambda x: pretreat.binning(x, deal_column=self.main_field,box_count=10, inplace=True))
        
    
    def plot(self,only_binned=False):
        if only_binned:
            self.binned_plot(only_binned)
        else:
            self.rankIC_plot()
            self.binned_plot()

    def rankIC_plot(self):
        fig = plt.figure(figsize=(1420/72/2,420/72))
        ax = fig.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        ax.yaxis.grid()
        # 高密度时 width 必须大于1，否则会显示不出来
        width_ = 0.5
        if len(self.rank_ic) > 500:
            width_ = 1
        plt.bar([pd.to_datetime(x).strftime('%Y%m%d') for x in self.rank_ic.index.values],self.rank_ic.fillna(0),width=width_)
        plt.title('rankIC', **PLOT_TITLE)
        plt.show()

        fig = plt.figure(figsize=(1420/72,420/72/7))
        ax = fig.gca()
        ax.xaxis.set_visible(False) 
        ax.yaxis.set_visible(False)
        ax.table(cellText=self.res.values.round(6),colLabels=self.res.columns,cellLoc='center', bbox = [0.0, 0.0, 1, 1]) 
        plt.title('desc', **PLOT_TITLE)
        plt.show()
        
        
    def get_ind_binned_ret_avg(self):
        # 此功能与 binned_plot 中，重复。
        ind_binned_noindex = self.ind_binned.reset_index().drop(['code', self.main_field],axis=1)
        return ind_binned_noindex.drop(['date'],axis=1).dropna().set_index('group_label').groupby(level=0).apply(lambda x: x['ret_forward'].sum())
    
    def get_ind_binned_ret_cumsum(self):
        # 此功能与 binned_plot 中，重复。
        ind_binned_noindex = self.ind_binned.reset_index().drop(['code', self.main_field],axis=1)
        ind_binned_ret_date = ind_binned_noindex.set_index(['date', 'group_label']).groupby(level=0).apply(lambda x: x.groupby(level=1).agg(sum))
        return ind_binned_ret_date.groupby(level=1).agg('cumsum')
    
        
    def binned_plot(self, only_binned=False):
        # 去除绘图不需要的原始因子和code
        ind_binned_noindex = self.ind_binned.reset_index().drop(['code', self.main_field],axis=1)
        # 按日期分组，组内再按分箱分组求总收益,结果会被倒序。
        ind_binned_ret_date = ind_binned_noindex.set_index(['date', 'group_label']).groupby(level=0).apply(lambda x: x.groupby(level=1).agg(sum))

        fig = plt.figure(figsize=(1420/72,320/72))
        ind_binned_ret_all = ind_binned_noindex.drop(['date'],axis=1).dropna().set_index('group_label').groupby(level=0).apply(lambda x: x['ret_forward'].sum())
        plt.bar(ind_binned_ret_all.index,ind_binned_ret_all)
        plt.title('分箱平均收益', **PLOT_TITLE)
        plt.show()
        
        if only_binned:
            return

        blenchmark = smpl.get_benchmark(name=self.sample, start=self.start, end=self.end, gap=self.gap)
        blenchmark_re = smpl.resample_stockdata_low(blenchmark.data,freq=self.freq)
        blenchmark_ret = smpl.get_forward_return(blenchmark_re,'close')
        blenchmark_ret.reset_index('code',drop=True,inplace=True)
        blenchmark_cum = blenchmark_ret.cumsum()

        fig = plt.figure(figsize=(1420/72,320/72))
        lns = ind_binned_ret_date.groupby(level=1).apply(lambda x: plt.plot(x.index.get_level_values(0).unique().tolist(),x.values.tolist(),label=x.index.get_level_values(1)[0]))
        ax2 = plt.gca().twinx()
        lns = [x[0] for x in lns.values] # lns,为了合并legend
        lns += ax2.plot(blenchmark_ret,linestyle=":", linewidth=2,color="black",label='bm')
        labs = [l.get_label() for l in lns]
        legend = plt.legend(lns, labs,loc='upper left',fontsize='x-small',title='反序\n注意\n10最小')
        legend.get_title().set_fontsize(fontsize = 12)
        plt.grid(linestyle="dotted",color="lightgray")
        plt.title('分箱收益变化', **PLOT_TITLE)
        plt.show()

        ind_binned_ret_cum = ind_binned_ret_date.groupby(level=1).apply(lambda x: x.cumsum())
        fig = plt.figure(figsize=(1420/72,320/72))
        lns = ind_binned_ret_cum.groupby(level=1).apply(lambda x: plt.plot(x.index.get_level_values(0).unique().tolist(),x.values.tolist(),label=x.index.get_level_values(1)[0]))
        ax3 = plt.gca().twinx()
        lns = [x[0] for x in lns.values] # lns,为了合并legend
        lns += ax3.plot(blenchmark_cum,linestyle=":", linewidth=2,color="black",label='bm')
        labs = [l.get_label() for l in lns]
        legend = plt.legend(lns, labs,loc='upper left',fontsize='x-small',title='反序\n注意\n10最小')
        legend.get_title().set_fontsize(fontsize = 12)
        plt.grid(linestyle="dotted",color="lightgray")
        plt.title('累计收益率', **PLOT_TITLE)
        plt.show()
