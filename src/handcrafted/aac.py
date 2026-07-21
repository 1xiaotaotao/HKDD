#!/usr/bin/env python
#_*_coding:utf-8_*_

"""ifeature code"""

# script for extract AAC features of protein

from collections import Counter
from src.utils.read_fasta_file import read_sequences
from .data import constants

alphabet = constants.ALPHABET


def aac(input_file):
	sequences = read_sequences(input_file)
	encodings = []
	for sequence in sequences:
		count = Counter(sequence)
		freq = {}
		for key in count:
			freq[key] = count[key] / len(sequence)
		
		code = []
		for a in alphabet:
			code.append(freq.get(a, 0))
		encodings.append(code)
	return encodings
