import numpy as np 
from bm3d import bm3d
from skimage.restoration import denoise_tv_chambolle

def bm3dPrior(m_, simul_params, pnp_params): 
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"]
    sigma = pnp_params["prior_sigma"]

    m = m_.reshape(ngy,ngx)
    m_denoise = bm3d(m, sigma)
    m_denoise = m_denoise.reshape(ngy*ngx, 1) 
    return m_denoise

def tvPrior(m_, simul_params): 
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"]
    m = m_.reshape(ngy, ngx) 
    m_denoise = denoise_tv_chambolle(m, weight = 0.4, multichannel=False) 
    m_denoise = m_denoise.reshape(ngy*ngx, 1)
    return m_denoise
