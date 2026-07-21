"""Pse-In-One Code"""

# script for extract Kmer DR and DT features of protein


"""
Created on Sat May 13 15:35:42 2016
@version:0.2.1./pyc
@author: Fule Liu, Nackel, luo
"""


from itertools import combinations_with_replacement, permutations, product
import numpy as np

from src.handcrafted.util import frequency
from src.utils.read_fasta_file import read_sequences
from .data import constants

alphabet = constants.ALPHABET
#===========================Kmer===================================================
def make_kmer_list(k):
	if k < 0:
		print('Error, k must be an inter and larger than 0.')

	kmers = []
	for i in range(1, k + 1):
		if len(kmers) == 0:
			kmers = list(alphabet)
		else:
			new_kmers = []
			for kmer in kmers:
				for c in alphabet:
					new_kmers.append(kmer + c)
			kmers = new_kmers

	return kmers
 
def make_kmer_vector(input_file, k=2):
	"""Generate kmer vector."""
	seq_list = read_sequences(input_file)
	
	vector = []
	kmer_list = make_kmer_list(k)
	for seq in seq_list:
		count_sum = 0
		
		# Generate the kmer frequency dict.
		kmer_count = {}
		for kmer in kmer_list:
			temp_count = frequency(seq, kmer)
			if kmer not in kmer_count:
				kmer_count[kmer] = 0
			kmer_count[kmer] += temp_count
			
			count_sum += temp_count
		
		# Normalize.
		count_vec = [kmer_count[kmer] for kmer in kmer_list]
		count_vec = [round(float(e) / count_sum, 8) for e in count_vec]
		vector.append(count_vec)
		

	return vector
#==============================================================================

#==============================DR for Protein==================================
def dr_method(input_file, max_dis=2):
	if int(max_dis) > 0:
		aa_pairs = make_kmer_list(2)
	else:
		aa_pairs = []
	aa_list = list(alphabet)
	vector_list = []
	seqs = read_sequences(input_file)
	for seq in seqs:
		vector = []
		seq = list(seq)
		len_seq = len(seq)
		for i in range(max_dis + 1):
			if i == 0:
				temp = [seq.count(j) for j in aa_list]
				vector.extend(temp)
			else:
				new_line = []
				for index, elem in enumerate(seq):
					if (index + i) < len_seq:
						new_line.append(seq[index] + seq[index + i])
				temp = [new_line.count(j) for j in aa_pairs]
				vector.extend(temp)
		vector_list.append(vector)
	return vector_list

#==============================================================================
#================================Distance Pair=================================

def get_pseaacdis_dict(raaas_lst, k):
	"""
	Args:
		raaas_lst: A list of reduced amino acid alphabet scheme.
		k: The length of pseudo amino acid composition.
	Returns:
		a pseaac_dis pattern dictionary.
	"""
	
	pseaacdis_lst = []
	pseaacdis_dict = {}
	if k == 2:
		part_pseaa = list(combinations_with_replacement(raaas_lst, 2))
		for element in part_pseaa:
			elelst = set(permutations(element, 2))
			pseaacdis_lst += elelst
		pseaacdis_lst.sort()

		for i in range(len(pseaacdis_lst)):
			a, b = pseaacdis_lst[i]
			for j in product(list(a), list(b)):
				pseaacdis_dict[j] = i
		return pseaacdis_dict
	elif k == 1:
		pseaacdis_lst = raaas_lst
		for i in range(len(pseaacdis_lst)):
			for j in list(pseaacdis_lst[i]):
				pseaacdis_dict[j] = i
		return pseaacdis_dict
	else:
		return False


def get_pseaacdis_vector_d(sequence, raaas_lst, distance):
	if distance == 0:
		pseaacdis_dict = get_pseaacdis_dict(raaas_lst, 1)
		sequence = list(sequence)
		vector = np.zeros((1, len(set(pseaacdis_dict.values()))))
		for i in sequence:
			position = pseaacdis_dict.get(i)
			vector[0, position] += 1
		return [round(f,3) for f in vector[0] / sum(vector[0])]
	elif distance > 0:
		pseaacdis_dict = get_pseaacdis_dict(raaas_lst, 2)
		sequence = list(sequence)
		vector = np.zeros((1, len(set(pseaacdis_dict.values()))))
		for i in range(len(sequence) - distance):
			a, b = sequence[i], sequence[i + distance]
			position = pseaacdis_dict.get((a, b))
			vector[0, position] += 1
		return [round(f, 3) for f in vector[0] / sum(vector[0])]
	else:
		return False


def get_pseaacdis_vector(sequence, raaas_lst, max_distance):
	vector = []
	if max_distance >= 0:
		for i in range(max_distance + 1):
			vector_tmp = get_pseaacdis_vector_d(sequence, raaas_lst, i)
			if i == 0:
				vector = vector_tmp
			else:
				vector = np.concatenate((vector_tmp, vector))
		return vector
	else:
		return False


def get_pseaacdis_matrix(input_file, reduce_alphabet_scheme=constants.CP_13, max_distance=1):
	seqs = read_sequences(input_file)
	features = []
	for seq in seqs:
		vector = get_pseaacdis_vector(seq, reduce_alphabet_scheme, max_distance)
		features.append(vector)
	return features
