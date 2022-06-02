import os
from easydict import EasyDict

CFG                             = EasyDict()
CFG.patch_num_points            = 256
CFG.up_ratio                    = 4
CFG.more_up                     = 2
CFG.patch_num_ratio             = 3
CFG.model_path                  = os.path.join(os.path.dirname(__file__), 'model')
CFG.gpu                         = '2'
CFG.progress_record             = True
