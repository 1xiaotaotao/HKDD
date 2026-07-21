from SaProt.model.saprot.base import SaprotBaseModel
from transformers import EsmTokenizer
from src.utils.read_fasta_file import read_sequences
import torch


from tqdm import tqdm

import logging

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class ExtractFeaturesSaProt:
	def __init__(self, pos_file, neg_file, model_path):
		self.input_pos_file = pos_file
		self.input_neg_file = neg_file
		self.model = None
		self.tokenizer = None
		self.model_path = str(model_path)
		self.config = {
			'task': 'base',
			'config_path': self.model_path,
			'load_pretrained': True,
		}
		self.batch_size = 8
		self.logger = logging.getLogger(self.__class__.__name__)
	
	def extract_process(self, sequences):
		model = self.model.to(device)
		features = []
		for i in tqdm(range(0, len(sequences), self.batch_size)):
			batch_sequences = sequences[i:i + self.batch_size]
			with torch.no_grad():
				inputs = self.tokenizer(batch_sequences, return_tensors='pt', padding=True, truncation=False,
				                        add_special_tokens=True)
				inputs = {k: v.to(device) for k, v in inputs.items()}
				embeddings = model.get_hidden_states(inputs, reduction='None')
			batch_features = torch.full((len(batch_sequences), 512, 1280), -1e9, dtype=torch.float, device=device)
			for j, seq in enumerate(batch_sequences):
				batch_features[j, 1:len(seq) // 2 + 1] = embeddings[j][:len(seq) // 2]
			features.append(batch_features.cpu())
		return torch.cat(features, dim=0)
	
	def extract(self):
		self.logger.info('extracting SaProt features')
		self.model = SaprotBaseModel(**self.config)
		self.tokenizer = EsmTokenizer.from_pretrained(self.model_path, local_files_only=True)
		pos_sequences = read_sequences(self.input_pos_file)
		neg_sequences = read_sequences(self.input_neg_file)
		self.logger.info('extracting features of pos by SaProt')
		pos_features = self.extract_process(pos_sequences)
		self.logger.info('extracting features of neg by SaProt')
		neg_features = self.extract_process(neg_sequences)
		features = torch.cat([pos_features, neg_features], dim=0)
		return features  # [n, 512, 1280]
