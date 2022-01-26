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
