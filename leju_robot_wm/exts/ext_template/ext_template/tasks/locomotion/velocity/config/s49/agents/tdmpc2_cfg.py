from omni.isaac.lab.utils import configclass

from .tdmpc2_cfg import KuavoS42TDMPC2RunnerCfg


@configclass
class KuavoS49TDMPC2RunnerCfg(KuavoS42TDMPC2RunnerCfg):
	"""TD-MPC2 runner config for Kuavo S49 (URDF, dance-only)."""

	def __post_init__(self):
		self.obs_profile = "dance"
		self.experiment_name = "Kuavo/s49/tdmpc2_dance"
		self.action_scale = 0.30
		self.batch_size = 96
		self.buffer_size = 200_000
		self.min_buffer_size = 384


@configclass
class KuavoS49TDMPC2PlayRunnerCfg(KuavoS49TDMPC2RunnerCfg):
	def __post_init__(self):
		super().__post_init__()
		self.mpc = True
		self.num_samples = 256
		self.num_elites = 32
