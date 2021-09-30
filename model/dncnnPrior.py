#-----------------------------------------
# Dncnn prior for CNN based image denoiser
#----------------------------------------

import os.path
import numpy as np
import torch

from util  import utils_model
from util  import utils_image as util 

def dncnnPrior( m_, channel_params, simul_params):
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"]
    
    lperm = channel_params["lperm"]
    hperm = channel_params["hperm"]

    m = m_.reshape(ngy, ngx)
    m = np.expand_dims(m, axis = 2) 
 
    model_name = 'dncnn_50'
    n_channels = 1
    nb = 17 
    model_pool = 'model_zoo'
    model_path = os.path.join(model_pool, model_name+'.pth')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from model.network_dncnn import DnCNN as net
    
    model = net(in_nc = 1, out_nc = n_channels, nc=64, nb = nb, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval() 
    model = model.to(device)
    m = util.single2tensor4(m)
    m = m.to(device) 
    m_denoise = utils_model.test_mode(model,m, mode = 3) 
    m_denoise = util.tensor2single(m_denoise)
    m_denoise = m_denoise.reshape(ngx*ngy, 1) 
    
    return m_denoise
    
