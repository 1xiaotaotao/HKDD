import torch
from copy import deepcopy
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torch.optim as optim
from .model import Model
from .dataset import AmyloidDataset
from ..utils.calculate_metrics import calculate_metrics, summarize_results
import numpy as np
from collections import defaultdict
import logging
from ..config import AmyloidConfig

deterministic_generator = torch.Generator()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class EarlyStopping:
	def __init__(self, patience=10, delta=0.01):
		self.patience = patience
		self.delta = delta
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.best_model_state = None
	
	def __call__(self, score, model):
		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(model)
		elif self.best_score > score + self.delta:
			self.best_score = score
			self.save_checkpoint(model)
			self.counter = 0
		else:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True
	
	def save_checkpoint(self, model):
		self.best_model_state = model.state_dict()
	
	def load_best_model(self, model):
		if self.best_model_state is not None:
			model.load_state_dict(self.best_model_state)


class Train:
	def __init__(self, dataset, config: AmyloidConfig, set_num=0):
		self.dataset = dataset
		self.config = config
		self.init_generator()
		train_pos_files = {'aa': f'data/aa/{self.dataset}/train_pos.fasta',
						   'sa': f'data/sa/{self.dataset}/train_pos.fasta'}
		train_neg_files = {'aa': f'data/aa/{self.dataset}/train_neg.fasta',
						   'sa': f'data/sa/{self.dataset}/train_neg.fasta'}
		test_pos_files = {'aa': f'data/aa/{self.dataset}/test_pos.fasta',
						  'sa': f'data/sa/{self.dataset}/test_pos.fasta'}
		test_neg_files = {'aa': f'data/aa/{self.dataset}/test_neg.fasta',
						  'sa': f'data/sa/{self.dataset}/test_neg.fasta'}
		self.train_set_path = f'dataset/{self.dataset}/train_set_{set_num}.pt'
		self.test_set_path = f'dataset/{self.dataset}/test_set_{set_num}.pt'
		self.train_set = AmyloidDataset.load_or_create(
			train_pos_files,
			train_neg_files,
			set_num,
			config,
			f'dataset/{dataset}/train_set_{set_num}.pt',
		)
		self.test_set = AmyloidDataset.load_or_create(
			test_pos_files,
			test_neg_files,
			set_num,
			config,
			f'dataset/{dataset}/test_set_{set_num}.pt',
		)
		self.training_config = config.training
		self.n_splits = 10
		self.model = Model(config, self.train_set.features_dim).to(device)
		self.epochs = self.training_config.epochs
		self.batch_size = self.training_config.batch_size
		self.criterion = nn.BCEWithLogitsLoss()
		self.early_stopping = EarlyStopping(patience=self.training_config.early_stopping_patience)
		self.attn_weights = None
		self.losses = None
		self.probs = None
		self.umap_features = None        
		self.best_model = None
		self.model_for_predict = None
		self.logger = logging.getLogger(self.__class__.__name__)
		
	def init_generator(self):
		deterministic_generator.manual_seed(self.config.seed)

	def model_params_path(self):
		model_path = Path(self.config.save.model_path)
		model_path.mkdir(parents=True, exist_ok=True)
		return model_path / 'model_params.pth'
	
	def train(self, model, train_loader, optimizer, scheduler):
		model.train()
		losses = []
		self.early_stopping = EarlyStopping()
		for _ in range(self.epochs):
			total_loss = 0
			for features, labels in train_loader:
				features = {name: value.to(device) for name, value in features.items()}
				labels = labels.to(device)
				optimizer.zero_grad()
				outputs = model(features)
				if isinstance(outputs, torch.Tensor):
					loss = self.criterion(outputs, labels)
				else:
					loss = torch.tensor(0.0, device=device)
					for key in outputs:
						if key == 'attn_weight' or key == 'umap_features':
							continue
						loss += outputs[key]['weight'] * self.criterion(outputs[key]['logit'], labels)
				total_loss += loss.item()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				optimizer.step()
			scheduler.step()
			avg_loss = total_loss / len(train_loader)
			losses.append(avg_loss)
			self.early_stopping(avg_loss, model)
			if self.early_stopping.early_stop:
				self.logger.info('early stopping')
				break
		self.losses = losses
	
	def eval(self, model, data_loader):
		model.eval()
		y_true, y_pred, y_pred_proba, attn_weights, umap_features = [], [], [], [], defaultdict(list)
		with torch.no_grad():
			for features, labels in data_loader:
				features = {name: value.to(device) for name, value in features.items()}
				labels = labels.to(device)
				outputs = model(features)
				if isinstance(outputs, dict):
					if 'attn_weight' in outputs:
						attn_weights.extend(outputs['attn_weight'].cpu().numpy())
					if 'umap_features' in outputs:
						for key, value in outputs['umap_features'].items():
							umap_features[key].extend(value.cpu().numpy())
					outputs = outputs['final']['logit']
				probabilities = torch.sigmoid(outputs)
				predictions = (probabilities > 0.5).float()
				y_true.extend(labels.cpu().numpy())
				y_pred.extend(predictions.cpu().numpy())
				y_pred_proba.extend(probabilities.cpu().numpy())
		metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
		self.attn_weights = attn_weights
		self.probs = y_pred_proba
		self.umap_features = umap_features
		self.logger.info(f'metrics:{metrics}')
		return metrics
	
	def cross_validation_on_train(self):
		dataset = self.train_set
		results = []
		best_acc = 0
		k_fold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.config.seed)
		split_generator = k_fold.split(range(len(self.train_set)), self.train_set.labels)
		self.logger.info(f'cross validation...')
		for fold, (train_idx, test_idx) in enumerate(split_generator):
			self.logger.info(f'fold {fold}')
			model = deepcopy(self.model).to(device)
			optimizer = optim.AdamW(model.parameters(), lr=self.training_config.init_learning_rate, weight_decay=0.01)
			scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
			train_dataset = Subset(dataset, train_idx)
			test_dataset = Subset(dataset, test_idx)
			train_loader = DataLoader(
				train_dataset,
				batch_size=self.batch_size,
				shuffle=True,
				generator=deterministic_generator
			)
			test_loader = DataLoader(
				test_dataset,
				batch_size=self.batch_size,
				shuffle=False,
			)
			self.train(model, train_loader, optimizer, scheduler)
			self.early_stopping.load_best_model(model)
			metrics = self.eval(model, test_loader)
			if metrics['accuracy'] > best_acc:
				best_acc = metrics['accuracy']
				self.best_model = deepcopy(model.cpu())
			results.append(metrics)
		summary_res = summarize_results(results)
		self.logger.info('mean results of 10 foldCV on train:')
		self.logger.info(summary_res)
		torch.save(self.best_model.state_dict(), self.model_params_path())
		return summary_res
	
	def train_on_set(self):
		self.logger.info('Training on the train set')
		model = deepcopy(self.model).to(device)
		train_dataset = self.train_set
		train_loader = DataLoader(
			train_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			generator=deterministic_generator
		)
		optimizer = optim.AdamW(model.parameters(), lr=self.training_config.init_learning_rate, weight_decay=0.01)
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
		self.train(model, train_loader, optimizer, scheduler)
		self.early_stopping.load_best_model(model)
		self.model_for_predict = deepcopy(model.cpu())
		torch.save(self.model_for_predict.state_dict(), self.model_params_path())
	
	def predict_on_test(self):
		self.logger.info('External validation')
		test_dataset = self.test_set
		test_loader = DataLoader(
			test_dataset,
			batch_size=self.batch_size,
			shuffle=False,
		)
		model = deepcopy(self.model).to(device)
		model.load_state_dict(torch.load(self.model_params_path()))
		return self.eval(model, test_loader)
	
	# return numpy arr
	def get_attn_weights(self):
		if not self.attn_weights:
			return None
		return np.mean(np.array(self.attn_weights), axis=0, keepdims=True)
	
	def get_losses(self):
		if not self.losses:
			return None
		return np.array(self.losses)
		
	def get_probs(self):
		if not self.probs:
			return None
		return np.array(self.probs)
	
	# return dict list: {'deep': [tensor_1, ..., tensor_last], ..., 'attn':[tensor_1, ..., tensor_last]}
	def get_umap_features(self):
		if not self.umap_features:
			return None
		return self.umap_features
