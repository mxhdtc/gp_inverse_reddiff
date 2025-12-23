
from algorithm import REDDIFF_VVGP
from .normal_vvgp import NORMAL_REG
from .algorithm_parallel import REDDIFF_VVGP_PARALLEL

def build_gp_algo(cg_model, forward_model, cfg):
	if cfg.algo.name == 'reddiff_vvgp':
		return REDDIFF_VVGP(cg_model, forward_model, cfg)
	elif cfg.algo.name == 'reddiff_vvgp_parallel':
		return REDDIFF_VVGP_PARALLEL(cg_model, forward_model, cfg)
	elif cfg.algo.name == 'normal_vvgp':
		return NORMAL_REG(cg_model, forward_model, cfg)
	else:
        		raise ValueError(f'No algorithm named {cfg.algo.name}')
