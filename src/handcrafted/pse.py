"""Pse-In-One Code"""

# script for extract PC-PseAAC(-General), SC-PseAAC(-General) features of protein

__author__ = 'Fule Liu'

import sys
import os
import pickle
from math import pow
from src.handcrafted.util import frequency, extra_aaindex, norm_index_vals
from src.utils.read_fasta_file import read_sequences
from src.handcrafted.nac import make_kmer_list


from .data import constants
current_dir = os.path.dirname(os.path.abspath(__file__))


class AAIndex:
	def __init__(self, head, index_dict):
		self.head = head
		self.index_dict = index_dict

	def __str__(self):
		return '%s\n%s' % (self.head, self.index_dict)
	
sys.modules['__main__'].AAIndex = AAIndex

alphabet = constants.ALPHABET



def get_phyche_list(phyche_list, all_prop=False):
	"""
		Get phyche_list and check it.
		Args:
			phyche_list: list, the input physicochemical properties list.
			all_prop: bool, choose all physicochemical properties or not.
		Returns:
			properties list
		"""
	if phyche_list is None or len(phyche_list) == 0:
		if not all_prop:
			error_info = 'Error, The phyche_list and all_prop can\'t be all False.'
			raise ValueError(error_info)
	all_prop_list = constants.PRO_LIST
	# Set and check physicochemical properties.
	try:
		# Set all properties.
		if all_prop:
			phyche_list = all_prop_list
		# Check phyche properties.
		else:
			for e in phyche_list:
				if e not in all_prop_list:
					error_info = 'Sorry, the physicochemical properties ' + e + ' is not exit.'
					raise NameError(error_info)
	except:
		raise

	return phyche_list



def get_aaindex(index_list):
	"""Get the aaindex from data/aaindex.data.

	:param index_list: the index we want to get.
	:return: a list of AAIndex obj.
	"""
	new_aaindex = []
	data_file = os.path.join(current_dir, 'data', 'aaindex.data')
	with open(data_file, 'rb') as f:
		aaindex = pickle.load(f)
		for index_vals in aaindex:
			if index_vals.head in index_list:
				new_aaindex.append(index_vals)

	return new_aaindex


def extend_aaindex(filename):
	"""Extend the user-defined AAIndex from user's file.
	:return: a list of AAIndex obj.
	"""
	aaindex = extra_aaindex(filename)
	for ind, e in enumerate(aaindex):
		aaindex[ind] = AAIndex(e.head, norm_index_vals(e.index_dict))

	return aaindex


def get_ext_ind_pro(filename):
	"""Get the extend indices from index file, only work for protein."""
	inds = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
	aaindex = []
	with open(filename, 'r') as f:
		lines = f.readlines()
		for i, line in enumerate(lines):
			if line[0] == '>':
				temp_name = line[1:].rstrip()
				vals = lines[i + 2].rstrip().split('\t')
				ind_val = {ind: float(val) for ind, val in zip(inds, vals)}
				aaindex.append(AAIndex(temp_name, ind_val))
	return aaindex


def pro_cor_fun1(ri, rj, aaindex_list):
	_sum = 0.0
	len_index = len(aaindex_list)
	for aaindex in aaindex_list:
		_sum += pow(aaindex.index_dict[ri] - aaindex.index_dict[rj], 2)
	return _sum / len_index


def pro_cor_fun2(ri, rj, aaindex):
	return aaindex.index_dict[ri] * aaindex.index_dict[rj]


def get_parallel_factor(k, lamada, sequence, phyche_value):
	"""Get the corresponding factor theta list."""
	theta = []
	l = len(sequence)

	for i in range(1, lamada + 1):
		temp_sum = 0.0
		for j in range(0, l - k - i + 1):
			nucleotide1 = sequence[j: j + k]
			nucleotide2 = sequence[j + i: j + i + k]
			temp_sum += pro_cor_fun1(nucleotide1, nucleotide2, phyche_value)
		theta.append(temp_sum / (l - k - i + 1))

	return theta


def get_series_factor(k, lamada, sequence, phyche_value):
	"""Get the corresponding series factor theta list."""
	theta = []
	l_seq = len(sequence)
	max_big_lamada = len(phyche_value)
	for small_lamada in range(1, lamada + 1):
		for big_lamada in range(max_big_lamada):
			temp_sum = 0.0
			for i in range(0, l_seq - k - small_lamada + 1):
				nucleotide1 = sequence[i: i + k]
				nucleotide2 = sequence[i + small_lamada: i + small_lamada + k]
				temp_sum += pro_cor_fun2(nucleotide1, nucleotide2, phyche_value[big_lamada])
			theta.append(temp_sum / (l_seq - k - small_lamada + 1))
	return theta


def make_pseknc_vector(sequence_list, phyche_value, k=1, w=0.05, lamada=1, method_type='PC-PseAAC'):
	"""Generate the pseknc vector."""
	kmer = make_kmer_list(k)
	vector = []

	for sequence in sequence_list:
		if len(sequence) < k or lamada + k > len(sequence):
			error_info = 'Sorry, the sequence length must be larger than ' + str(lamada + k)
			sys.stderr.write(error_info)
			sys.exit(0)

		# Get the nucleotide frequency in the DNA sequence.
		fre_list = [frequency(sequence, str(key)) for key in kmer]
		fre_sum = float(sum(fre_list))

		# Get the normalized occurrence frequency of nucleotide in the DNA sequence.
		fre_list = [e / fre_sum for e in fre_list]
		theta_list = []
		if 'PC-PseAAC' == method_type:
			theta_list = get_parallel_factor(k, lamada, sequence, phyche_value)
		elif 'SC-PseAAC' == method_type:
			theta_list = get_series_factor(k, lamada, sequence, phyche_value)
		theta_sum = sum(theta_list)

		# Generate the vector according the Equation 9.
		denominator = 1 + w * theta_sum

		temp_vec = [round(f / denominator, 8) for f in fre_list]
		for theta in theta_list:
			temp_vec.append(round(w * theta / denominator, 8))

		vector.append(temp_vec)

	return vector


def pseknc(input_data, k=1, w=0.1, lamada=10, phyche_list=None, extra_index_file=None, all_prop=False, method_type='PC-PseAAC'):
	"""
		This is a complete acc in PseKNC.
		Args:
			input_data: fasta file, input by user.
			k: int, determines the length of the basic unit for feature extraction.
			w: int, weight factor that controls the contribution of sequence-order information.
			lamada:int, represents the highest rank (tier) of sequence-order correlation.
			phyche_list: list, the input physicochemical properties list.
			extra_index_file: a file path includes the user-defined phyche_index.
			all_prop: bool, choose all physicochemical properties or not.
			method_type: select ac, cc or acc to extract.

		Returns:
			a list of designated feature
		"""
	if phyche_list is None:
		phyche_list = ['Hydrophobicity', 'Hydrophilicity', 'Mass']
	phyche_list = get_phyche_list(phyche_list, all_prop=all_prop)
	# Get phyche_vals.
	phyche_vals = get_aaindex(phyche_list)
	if extra_index_file is not None:
		phyche_vals.extend(extend_aaindex(extra_index_file))
	seq_list = read_sequences(input_data)

	return make_pseknc_vector(seq_list, phyche_vals, k, w, lamada, method_type)



	# # Test protein.
	# default_pro = ['Hydrophobicity', 'Hydrophilicity', 'Mass']
	# alphabet = index_list.PROTEIN
	# res = pseknc(input_data=open('aa/test_pro.fasta'), k=1, w=0.05, lamada=2,
	#              phyche_list=['Hydrophobicity', 'Hydrophilicity'], extra_index_file="aa/test_ext_pro.txt",
	#              alphabet=alphabet, theta_type=1)
	#
	# for e in res:
	#     print(len(e), e)
