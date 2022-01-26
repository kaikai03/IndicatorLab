def check_MA_above_multi(hy_tuple, self_obj):
    """计算均线以上的数量(for多进程)
       :param hy_tuple: (bloc_kname, hy_codes)
    """
    import QUANTAXIS as QA
    import Ind_Ponding_VIX
    import dill
    import pandas as pd
    import numpy as np
    from base.Constants import LOW_FREQUENCE
    
    self_ = dill.loads(self_obj)
    
    hy_codes = hy_tuple[1]
    bloc_kname = hy_tuple[0]
    
    try:
        if self_.frequence == QA.FREQUENCE.DAY:
            data = QA.QA_fetch_stock_day_adv(hy_codes, self_.start, self_.end)
        else:
            data = QA.QA_fetch_stock_min_adv(hy_codes, self_.start, self_.end,frequence=self_.frequence)
        data = data.data
#             data = data.to_hfq().data
    except Exception as e:
        print(e)
        return []

    if data is None:
        return []

    pr = Ind_Ponding_VIX.PondingRate(data)
    pr.change_pramas(window=20)
    pr.set_ignore_sub_ind(True)
    pr.fit()

    ind = pr.ind_df['main']
    count = len(hy_codes)
    compared = ind[(self_.MA-1)*count:] > ind.groupby(level=1).apply(lambda x: QA.MA(x, self_.MA))[(self_.MA-1)*count:]
    if self_.frequence in LOW_FREQUENCE:
        date_label = 'date'
    else:
        date_label = 'datetime'
    res = compared.groupby(level=0).apply(lambda x: (x.index.get_level_values(date_label)[0], round(np.sum(x)/len(x),2)))   
    p = pd.DataFrame({date_label:[i[0] for i in res],'-':[i[1] for i in res]})
    return p.assign(blockname=bloc_kname)
