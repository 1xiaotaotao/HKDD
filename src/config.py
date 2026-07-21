from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def _value(text):
	text = text.strip()
	if text.lower() in ('true', 'false'):
		return text.lower() == 'true'
	try:
		return int(text)
	except ValueError:
		try:
			return float(text)
		except ValueError:
			return text.strip('\'\"')


def _simple_yaml(text):
	'''Small fallback parser for this project's two-level config file.'''
	config = {}
	section = config
	for raw_line in text.splitlines():
		line = raw_line.split('#', 1)[0].rstrip()
		if not line.strip():
			continue
		indent = len(line) - len(line.lstrip())
		key, value = line.strip().split(':', 1)
		if indent == 0:
			if value.strip():
				config[key] = _value(value)
			else:
				config[key] = {}
				section = config[key]
		else:
			section[key] = _value(value)
	return config


def _validate_fields(config, expected_fields, section):
	if not isinstance(config, Mapping):
		raise ValueError(f'{section} must be a mapping')
	missing_fields = sorted(expected_fields - config.keys())
	if missing_fields:
		missing_field_list = ', '.join(missing_fields)
		raise ValueError(f'{section} is missing required fields: {missing_field_list}')
	unknown_fields = sorted(config.keys() - expected_fields)
	if unknown_fields:
		unknown_field_list = ', '.join(unknown_fields)
		raise ValueError(f'{section} contains unknown fields: {unknown_field_list}')


def _validate_choice(name, value, choices):
	if not isinstance(value, str) or value not in choices:
		allowed_values = ', '.join(sorted(choices))
		raise ValueError(f'{name} must be one of: {allowed_values}; got {value!r}')


def _validate_positive_integer(name, value):
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		raise ValueError(f'{name} must be a positive integer; got {value!r}')

def _validate_not_negative_integer(name, value):
	if not isinstance(value, int) or isinstance(value, bool) or value < 0:
		raise ValueError(f'{name} must be a integer >= 0; got {value!r}')

def _validate_boolean(name, value):
	if not isinstance(value, bool):
		raise ValueError(f'{name} must be a boolean; got {value!r}')


def _validate_path(name, value):
	if not isinstance(value, (str, Path)):
		raise ValueError(f'{name} must be a path; got {value!r}')
	path_text = str(value)
	if not path_text.strip() or '\x00' in path_text:
		raise ValueError(f'{name} must be a non-empty path; got {value!r}')
	return Path(path_text)


@dataclass
class PretrainedModelsConfig:
	t5_path: Path
	bert_path: Path
	saprot_path: Path

	def __post_init__(self):
		self.t5_path = _validate_path('pretrained_models.t5_path', self.t5_path)
		self.bert_path = _validate_path('pretrained_models.bert_path', self.bert_path)
		self.saprot_path = _validate_path('pretrained_models.saprot_path', self.saprot_path)


@dataclass
class SaveConfig:
	model_path: Path
	attn_weights_path: Path
	losses_path: Path
	probs_path: Path
	umap_features_path: Path

	def __post_init__(self):
		self.model_path = _validate_path('save.model_path', self.model_path)
		self.attn_weights_path = _validate_path(
			'save.attn_weights_path',
			self.attn_weights_path,
		)
		self.losses_path = _validate_path('save.losses_path', self.losses_path)
		self.probs_path = _validate_path('save.probs_path', self.probs_path)
		self.umap_features_path = _validate_path(
			'save.umap_features_path',
			self.umap_features_path,
		)

	def initialize(self, dataset, seed):
		path_suffix = Path(dataset) / str(seed)
		self.model_path = self.model_path / path_suffix
		self.attn_weights_path = self.attn_weights_path / path_suffix
		self.losses_path = self.losses_path / path_suffix
		self.probs_path = self.probs_path / path_suffix
		self.umap_features_path = self.umap_features_path / path_suffix


@dataclass
class ModelConfig:
	feature_mode: str
	use_attention: bool
	deep_feature_mode: str
	semantic_feature_extractor: str
	deep_model: str
	joint_loss: bool

	def __post_init__(self):
		_validate_choice('model.feature_mode', self.feature_mode, {'know', 'deep', 'fusion'})
		_validate_boolean('model.use_attention', self.use_attention)
		_validate_choice(
			'model.deep_feature_mode',
			self.deep_feature_mode,
			{'semantic', 'structure', 'combined'},
		)
		_validate_choice(
			'model.semantic_feature_extractor',
			self.semantic_feature_extractor,
			{'prot-t5', 'esm-1b', 'esm-2', 'prot-bert'},
		)
		_validate_choice(
			'model.deep_model',
			self.deep_model,
			{'ffn', 'mamba', 'lstm', 'transformer'},
		)
		_validate_boolean('model.joint_loss', self.joint_loss)


@dataclass
class TrainingConfig:
	batch_size: int
	init_learning_rate: float
	epochs: int
	early_stopping_patience: int

	def __post_init__(self):
		_validate_positive_integer('training.batch_size', self.batch_size)
		if (
			not isinstance(self.init_learning_rate, (int, float))
			or isinstance(self.init_learning_rate, bool)
			or self.init_learning_rate <= 0
		):
			raise ValueError(
				'training.init_learning_rate must be greater than 0; '
				f'got {self.init_learning_rate!r}'
			)
		_validate_positive_integer('training.epochs', self.epochs)
		_validate_positive_integer(
			'training.early_stopping_patience', self.early_stopping_patience
		)



@dataclass
class AmyloidConfig:
	dataset: str
	seed: int
	pretrained_models: PretrainedModelsConfig
	model: ModelConfig
	training: TrainingConfig
	save: SaveConfig

	def __post_init__(self):
		_validate_choice('dataset', self.dataset, {'Amy', 'Cross_Beta_DB'})
		_validate_not_negative_integer('seed', self.seed)
		if not isinstance(self.pretrained_models, PretrainedModelsConfig):
			raise ValueError('pretrained_models must be a PretrainedModelsConfig')
		if not isinstance(self.model, ModelConfig):
			raise ValueError('model must be a ModelConfig')
		if not isinstance(self.training, TrainingConfig):
			raise ValueError('training must be a TrainingConfig')
		if not isinstance(self.save, SaveConfig):
			raise ValueError('save must be a SaveConfig')
		self.save.initialize(self.dataset, self.seed)

	@classmethod
	def from_mapping(cls, config):
		_validate_fields(
			config,
			{'dataset', 'seed', 'pretrained_models', 'model', 'training', 'save'},
			'config',
		)
		pretrained_models = config['pretrained_models']
		_validate_fields(
			pretrained_models,
			{'t5_path', 'bert_path', 'saprot_path'},
			'pretrained_models',
		)
		model = config['model']
		_validate_fields(
			model,
			{
				'feature_mode',
				'use_attention',
				'deep_feature_mode',
				'semantic_feature_extractor',
				'deep_model',
				'joint_loss',
			},
			'model',
		)
		training = config['training']
		_validate_fields(
			training,
			{
				'batch_size',
				'init_learning_rate',
				'epochs',
				'early_stopping_patience',
			},
			'training',
		)
		save = config['save']
		_validate_fields(
			save,
			{
				'model_path',
				'attn_weights_path',
				'losses_path',
				'probs_path',
				'umap_features_path',
			},
			'save',
		)
		return cls(
			dataset=config['dataset'],
			seed=config['seed'],
			pretrained_models=PretrainedModelsConfig(**pretrained_models),
			model=ModelConfig(**model),
			training=TrainingConfig(**training),
			save=SaveConfig(**save),
		)


def load_config(path: str | Path = 'config.yaml') -> AmyloidConfig:
	text = Path(path).read_text(encoding='utf-8')
	try:
		import yaml
		config = yaml.safe_load(text)
	except ImportError:
		config = _simple_yaml(text)
	return AmyloidConfig.from_mapping(config)
