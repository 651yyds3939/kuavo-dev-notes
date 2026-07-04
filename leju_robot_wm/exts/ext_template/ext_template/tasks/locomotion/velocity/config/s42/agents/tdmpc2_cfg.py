from omni.isaac.lab.utils import configclass


@configclass
class KuavoS42TDMPC2RunnerCfg:
	"""TD-MPC2 runner config for Kuavo S42/S46 (stable USD asset)."""

	seed: int = 42
	device: str = "cuda:0"
	experiment_name: str = "Kuavo/s42/tdmpc2"
	run_name: str = ""
	max_iterations: int = 5000
	save_interval: int = 200
	log_interval: int = 10

	# obs / action
	obs_profile: str = "dance"
	action_scale: float = 0.30

	# collection — updates_per_iter<=0 enables auto (transitions_per_iter // batch_size)
	steps_per_env: int = 24
	seed_steps: int = 0
	seed_iterations: int = 25
	min_buffer_size: int = 512
	updates_per_iter: int = 0
	max_updates_per_iter: int = 384

	# TD-MPC2 model (8GB-friendly defaults)
	learning_rate: float = 3e-4
	batch_size: int = 128
	horizon: int = 5
	sequence_length: int = 50
	buffer_size: int = 300_000
	latent_dim: int = 512
	mpc: bool = False
	num_samples: int = 256
	num_elites: int = 32
	num_pi_trajs: int = 12

	# runner behaviour
	init_at_random_ep_len: bool = True

	# resume
	resume: bool = False
	load_run: str = ""
	load_checkpoint: str = ""


@configclass
class KuavoS42StandTDMPC2RunnerCfg(KuavoS42TDMPC2RunnerCfg):
	"""WM stage 1: stand still (87-dim velocity obs profile)."""

	def __post_init__(self):
		self.obs_profile = "velocity"
		self.experiment_name = "Kuavo/s42/tdmpc2_stand"
		self.action_scale = 0.15
		self.seed_iterations = 30
		self.max_iterations = 3000
		self.init_at_random_ep_len = False


@configclass
class KuavoS42StandTDMPC2PlayRunnerCfg(KuavoS42StandTDMPC2RunnerCfg):
	def __post_init__(self):
		super().__post_init__()
		self.mpc = False
		self.num_samples = 512
		self.num_elites = 64


@configclass
class KuavoS42VelocityTDMPC2RunnerCfg(KuavoS42TDMPC2RunnerCfg):
	"""WM stage 2: velocity tracking — resume from stand checkpoint (same 87-dim obs)."""

	def __post_init__(self):
		self.obs_profile = "velocity"
		self.experiment_name = "Kuavo/s42/tdmpc2_velocity"
		self.action_scale = 0.20
		self.seed_iterations = 15
		self.max_iterations = 5000


@configclass
class KuavoS42VelocityTDMPC2PlayRunnerCfg(KuavoS42VelocityTDMPC2RunnerCfg):
	def __post_init__(self):
		super().__post_init__()
		self.mpc = False
		self.num_samples = 512
		self.num_elites = 64


@configclass
class KuavoS42DanceTDMPC2RunnerCfg(KuavoS42TDMPC2RunnerCfg):
	"""WM stage 4: full-body dance (115-dim obs). Train after arms-only stage."""

	def __post_init__(self):
		self.obs_profile = "dance"
		self.experiment_name = "Kuavo/s42/tdmpc2_dance"
		self.action_scale = 0.15
		self.seed_iterations = 25
		self.max_iterations = 8000
		self.init_at_random_ep_len = False


@configclass
class KuavoS42ArmsOnlyTDMPC2RunnerCfg(KuavoS42DanceTDMPC2RunnerCfg):
	"""WM stage 3: arms-only dance while legs stay near default."""

	def __post_init__(self):
		super().__post_init__()
		self.experiment_name = "Kuavo/s42/tdmpc2_dance_arms"
		self.action_scale = 0.15
		self.max_iterations = 4000


@configclass
class KuavoS42TDMPC2PlayRunnerCfg(KuavoS42DanceTDMPC2RunnerCfg):
	def __post_init__(self):
		super().__post_init__()
		self.mpc = False
		self.num_samples = 512
		self.num_elites = 64
