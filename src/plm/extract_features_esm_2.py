from src.utils.read_fasta_file import read_sequences_truncated

import esm
import torch
from tqdm import tqdm
import logging


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
class ExtractFeaturesEsm2:
	def __init__(self, pos_file, neg_file):
		self.input_pos_file = pos_file
		self.input_neg_file = neg_file
		self.model = None
		self.alphabet = None
		self.batch_size = 8
		self.logger = logging.getLogger(self.__class__.__name__)
	
		
	def extract_process(self, sequences):
		model = self.model.to(device)
		converter = self.alphabet.get_batch_converter()
		features = []
		for i in tqdm(range(0, len(sequences), self.batch_size)):
			batch_sequences = sequences[i:i + self.batch_size]
			batch_data = [('', seq) for seq in batch_sequences]
			_, _, tokens = converter(batch_data)
			tokens = tokens.to(device)
			with torch.no_grad():
				results = model(tokens, repr_layers=[33])
			token_representations = results['representations'][33]
			batch_features = torch.full((len(batch_sequences), 512, 1280), -1e9, dtype=torch.float, device=device)
			for j, seq in enumerate(batch_sequences):
				batch_features[j, 1:len(seq) + 1] = token_representations[j, 1:len(seq) + 1]
			features.append(batch_features.cpu())
		return torch.cat(features, dim=0)
	
		
	
	def extract(self):
		self.logger.info('extracting esm2 features')
		self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
		self.model.eval()
		pos_sequences = read_sequences_truncated(self.input_pos_file)
		neg_sequences = read_sequences_truncated(self.input_neg_file)
		self.logger.info('extracting features of pos by esm2')
		pos_features = self.extract_process(pos_sequences)
		self.logger.info('extracting features of neg by esm2')
		neg_features = self.extract_process(neg_sequences)
		return torch.cat([pos_features, neg_features], dim=0)  # [n, 512, 1024] #[n, 512 ,1280]
