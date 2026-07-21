import torch
from tqdm import tqdm
import re
from transformers import BertTokenizer, BertModel
from src.utils.read_fasta_file import read_sequences_truncated
import logging

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class ExtractFeaturesBert:
	def __init__(self, input_pos_file, input_neg_file, model_path):
		self.input_pos_file = input_pos_file
		self.input_neg_file = input_neg_file
		self.model_path = str(model_path)
		self.model = None
		self.tokenizer = None
		self.batch_size = 8
		self.logger = logging.getLogger(self.__class__.__name__)
	
	def extract_process(self, sequences):
		model = self.model.to(device)
		features = []
		for i in tqdm(range(0, len(sequences), self.batch_size)):
			batch_sequences = sequences[i:i + self.batch_size]
			filled_sequences = [' '.join(list(re.sub(r'[UZOB]', 'X', str(sequence)))) for sequence in batch_sequences]
			output = self.tokenizer.batch_encode_plus(filled_sequences, add_special_tokens=True, padding='longest',
			                                          return_tensors='pt')
			ids, attention_mask = output['input_ids'].to(device), output['attention_mask'].to(device)
			with torch.no_grad():
				result = model(ids, attention_mask)
			embedding = result.last_hidden_state
			batch_features = torch.full((len(batch_sequences), 512, 1024), -1e9, dtype=torch.float, device=device)
			for j, seq in enumerate(batch_sequences):
				batch_features[j, 1:len(seq) + 1] = embedding[j, 1:len(seq) + 1]
			features.append(batch_features.cpu())
		return torch.cat(features, dim=0)
	
	def extract(self):
		self.logger.info('extracting Bert features')
		self.tokenizer = BertTokenizer.from_pretrained(self.model_path, local_files_only=True, legacy=False)
		self.model = BertModel.from_pretrained(self.model_path, local_files_only=True)
		self.model.eval()
		pos_sequences = read_sequences_truncated(self.input_pos_file)
		neg_sequences = read_sequences_truncated(self.input_neg_file)
		self.logger.info('extracting features of pos by bert')
		pos_features = self.extract_process(pos_sequences)
		self.logger.info('extracting features of neg by bert')
		neg_features = self.extract_process(neg_sequences)
		return torch.cat([pos_features, neg_features], dim=0)  # [n, 512, 1024]