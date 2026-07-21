"""Pse-In-One Code"""

# script for extract AC CC ACC features of protein

__author__ = 'Fule Liu'

from src.utils.read_fasta_file import read_sequences
from src.handcrafted.pse import get_phyche_list, get_aaindex, extend_aaindex
from .data import constants
import logging
logger = logging.getLogger(__name__)

alphabet = constants.ALPHABET

def acc(input_data, k=1, lag=2, phyche_list=None, extra_index_file=None, all_prop=False, method_type='ACC'):
	"""
	This is a complete acc in PseKNC.
	Args:
		input_data: fasta file, input by user.
		k: int, determines the length of the basic unit for feature extraction.
		lag: int, determines the distance for correlation calculation.
		phyche_list: list, the input physicochemical properties list.
		extra_index_file: a file path includes the user-defined phyche_index.
		all_prop: bool, choose all physicochemical properties or not.
		method_type: select ac, cc or acc to extract.
		
	Returns:
		list, a list of designated feature
	"""
	
	if phyche_list is None:
		phyche_list = ['Hydrophobicity', 'Hydrophilicity', 'Mass']
	phyche_list = get_phyche_list(phyche_list, all_prop=all_prop)
	# print(phyche_list)
	# Get phyche_vals.
	phyche_vals = get_aaindex(phyche_list)
	# print(phyche_vals)
	if extra_index_file is not None:
		phyche_vals.extend(extend_aaindex(extra_index_file))
	seqs = read_sequences(input_data)
	# Transform the aa format to dict {acid: [phyche_vals]}.
	phyche_keys = phyche_vals[0].index_dict.keys()
	phyche_vals = [e.index_dict.values() for e in phyche_vals]
	new_phyche_vals = zip(*[e for e in phyche_vals])
	phyche_vals = {key: list(val) for key, val in zip(phyche_keys, new_phyche_vals)}
	if method_type == 'AC':
		return make_ac_vec(seqs, lag, phyche_vals, k)
	elif method_type == 'CC':
		return make_cc_vec(seqs, lag, phyche_vals, k)
	else:
		return make_acc_vec(seqs, lag, phyche_vals, k)


def make_ac_vec(sequence_list, lag, phyche_value, k):
	# Get the length of phyche_vals.
	phyche_values = list(phyche_value.values())
	len_phyche_value = len(phyche_values[0])

	vec_ac = []
	for sequence in sequence_list:
		len_seq = len(sequence)
		each_vec = []

		for temp_lag in range(1, lag + 1):
			for j in range(len_phyche_value):

				# Calculate average phyche_value for a nucleotide.
				ave_phyche_value = 0.0
				for i in range(len_seq - temp_lag - k + 1):
					nucleotide = sequence[i: i + k]
					ave_phyche_value += float(phyche_value[nucleotide][j])
				ave_phyche_value /= len_seq

				# Calculate the vector.
				temp_sum = 0.0
				for i in range(len_seq - temp_lag - k + 1):
					nucleotide1 = sequence[i: i + k]
					nucleotide2 = sequence[i + temp_lag: i + temp_lag + k]
					temp_sum += (float(phyche_value[nucleotide1][j]) - ave_phyche_value) * (
						float(phyche_value[nucleotide2][j]))

				each_vec.append(round(temp_sum / (len_seq - temp_lag - k + 1), 8))
		vec_ac.append(each_vec)

	return vec_ac


def make_cc_vec(sequence_list, lag, phyche_value, k):
	phyche_values = list(phyche_value.values())
	len_phyche_value = len(phyche_values[0])

	vec_cc = []
	for sequence in sequence_list:
		len_seq = len(sequence)
		each_vec = []

		for temp_lag in range(1, lag + 1):
			for i1 in range(len_phyche_value):
				for i2 in range(len_phyche_value):
					if i1 != i2:
						# Calculate average phyche_value for a nucleotide.
						ave_phyche_value1 = 0.0
						ave_phyche_value2 = 0.0
						for j in range(len_seq - temp_lag - k + 1):
							nucleotide = sequence[j: j + k]
							ave_phyche_value1 += float(phyche_value[nucleotide][i1])
							ave_phyche_value2 += float(phyche_value[nucleotide][i2])
						ave_phyche_value1 /= len_seq
						ave_phyche_value2 /= len_seq

						# Calculate the vector.
						temp_sum = 0.0
						for j in range(len_seq - temp_lag - k + 1):
							nucleotide1 = sequence[j: j + k]
							nucleotide2 = sequence[j + temp_lag: j + temp_lag + k]
							temp_sum += (float(phyche_value[nucleotide1][i1]) - ave_phyche_value1) * \
										(float(phyche_value[nucleotide2][i2]) - ave_phyche_value2)
						each_vec.append(round(temp_sum / (len_seq - temp_lag - k + 1), 8))

		vec_cc.append(each_vec)

	return vec_cc


def make_acc_vec(seqs, lag, phyche_values, k):
	from functools import reduce
	zipped = list(zip(make_ac_vec(seqs, lag, phyche_values, k), make_cc_vec(seqs, lag, phyche_values, k)))
	return [reduce(lambda x, y: x + y, e) for e in zipped]

# Test ACC for PROTEIN.
# print("Test ACC for PROTEIN.")
# res = acc(open(''), k=1, lag=2, theta_type=3,
#           phyche_list=['Hydrophobicity', 'Hydrophilicity', 'Mass'])
# print(len(res[0]), res)
