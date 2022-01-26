
import sys
import os 
module_path = os.path.abspath(os.path.join('..')) 
if module_path not in sys.path: 
    sys.path.append(module_path)
    
import pandas as pd
import copy

import QUANTAXIS as QA


from base.Constants import LOW_FREQUENCE
from QAIndicatorStructExt import QA_DataStruct_Indicators_Ext

class Ind_Model:
    # （静态）最优描述符号，指标人工训练结束后的记录，用于给后续更大的模型引用时做提示
#     optimum_param={'valid':False, 'main':'main', 'desition_direct':1, 'window':14, 'freq':'d','neutralize':{'enable':False,'static_mv':False}}

    def __init__(self,data_df, ind_name, frequence=QA.FREQUENCE.DAY, pramas_default=None):
        """
        注意：默认是 fast_mode 的，为了批量生成因子的时候加速。
        """
#         if not isinstance(data, type(QA.OUTPUT_FORMAT.DATASTRUCT)):
#             raise TypeError('Must be DATASTRUCT')
        if pramas_default is None:
            self._pramas_default = self.on_set_params_default()
        else:
            self._pramas_default = pramas_default
            
        if not isinstance(self._pramas_default, dict):
            raise TypeError('_pramas_default MUST BE DICT')
        
        self.fast_mode = True
        self.pramas = copy.deepcopy(self._pramas_default)
        self.ind_name = ind_name
        self.data = data_df
        self.ind_df = None
        self.desition_df = None
        self.frequence = frequence
        

    def __repr__(self):
        return '< {} in pramas  {} ,{},{} >'.format(self.ind_name, 
                                                    str (self.pramas), 
                                                    'not fit' if not self.is_fitted else 'fitted:'+ str(self.keys_ind), 
                                                    'not desition' if not self.has_desition else 'has desition')
        
    def change_pramas(self,**dic):
        for k in dic.keys():
            self.pramas[k] = dic[k]
    
    def reset_pramas(self):
        self.pramas = copy.deepcopy(self._pramas_default)
    
    @property
    def cur_pramas(self):
        return self.pramas
    @property
    def default_pramas(self):
        return self._pramas_default
    @property
    def keys_pramas(self):
        return self._pramas_default.keys()
    @property
    def keys_ind(self):
        if not self.is_fitted:
            raise Exception("need fit the model first")
        return self.ind_df.keys().to_list()
    
    @property
    def is_fitted(self):
        return not self.ind_df is None
    
    @property
    def has_desition(self):
        return not self.desition_df is None
    
    @property
    def is_low_frequence(self):
        return self.frequence in LOW_FREQUENCE
    

    @property
    def ind(self):
        if not self.is_fitted:
            raise Exception("need fit the model first")
        return QA_DataStruct_Indicators_Ext(self.ind_df)

    @property
    def desition(self):
        if not self.has_desition  or not isinstance(self.desition_df, pd.DataFrame):
            print("desition_df:",self.desition_df)
            raise Exception('on_desition_structure error')
        return QA_DataStruct_Indicators_Ext(self.desition_df)
    
    def set_fast_mode(self, is_fast = True):
        self.fast_mode = is_fast
    

    def on_set_params_default(self) -> dict:
        #初始化时设置默认参数
        raise NotImplementedError
        
    def on_indicator_structuring(self, data) -> pd.DataFrame:
        #构造因子
        raise NotImplementedError
        
    def on_desition_structuring(self, data, ind_data) -> pd.DataFrame:
        #生成因子结论，非必要
        raise NotImplementedError
        
        
        
    def excute_for_multicode(self, data, func, **pramas):
        return data.groupby(level=1, as_index=False, group_keys=False).apply(func,**pramas)

    def fit(self):
        self.ind_df = self.on_indicator_structuring(self.data)
        
        try:
            self.desition_df = self.on_desition_structuring(self.data, self.ind_df)
        except NotImplementedError:
            self.desition_df = None
        except Exception as err:
            raise Exception(err)
            
    def checking(self,benchmark=None):
        print('evaluation table')
        
