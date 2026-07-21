import logging
from functools import partial
import torch
from sklearn.preprocessing import StandardScaler
from src.handcrafted import *
import numpy as np

class ExtractFeaturesHandcrafted:
	def __init__(self, pos_file, neg_file):
		self.input_pos_file = pos_file
		self.input_neg_file = neg_file
		self.feature_functions = {
			'AAC': aac,
			'CTDC': ctdc,
			'CTDT': ctdt,
			'CTDD': ctdd,
			'ACC': acc,
			'PC_PseAAC': partial(pseknc, method_type='PC-PseAAC'),
			'SC_PseAAC': partial(pseknc, method_type='SC-PseAAC'),
			'Kmer': make_kmer_vector,
			'DR': dr_method,
			'DP': get_pseaacdis_matrix,
		}
		self.scaler = StandardScaler()
		self.logger = logging.getLogger(self.__class__.__name__)
		
	def extract_features(self, feature):
		self.logger.info(f'Extracting features from {feature}')
		feature_func = self.feature_functions[feature]
		pos_features_data_list = feature_func(self.input_pos_file)
		neg_features_data_list = feature_func(self.input_neg_file)
		return pos_features_data_list + neg_features_data_list
	
	def extract(self):
		aac = np.array(self.extract_features('AAC'), dtype=np.float32)
		acc = np.array(self.extract_features('ACC'), dtype=np.float32)
		ctdc = np.array(self.extract_features('CTDC'), dtype=np.float32)
		ctdd = np.array(self.extract_features('CTDD'), dtype=np.float32)
		ctdt = np.array(self.extract_features('CTDT'), dtype=np.float32)
		pc_pseaac = np.array(self.extract_features('PC_PseAAC'), dtype=np.float32)
		sc_pseaac = np.array(self.extract_features('SC_PseAAC'), dtype=np.float32)
		# kmer = np.array(self.extract_features('Kmer'), dtype=np.float32)
		dr = np.array(self.extract_features('DR'), dtype=np.float32)
		dp = np.array(self.extract_features('DP'), dtype=np.float32)
		combined = np.hstack([aac, acc, ctdc, ctdd, ctdt, pc_pseaac, sc_pseaac, dr, dp])
		combined = StandardScaler().fit_transform(combined)
		return torch.from_numpy(combined).cpu() # (n, 1393)
	