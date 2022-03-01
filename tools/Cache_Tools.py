
import sys
import os 
module_path = os.path.abspath(os.path.join('..')) 
if module_path not in sys.path: 
    sys.path.append(module_path)

import pandas as pd
import numpy as np

import tools.Sample_Tools as smpl
from base.JuUnits import get_root_path

__cache_dir__ = get_root_path()+'/cache.feather/'


df = smpl.get_sample_by_zs(name='上证50', end='2021-11-28', gap=250,  only_main=True, filter_st=True).data
df
