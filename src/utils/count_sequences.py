from Bio import SeqIO


def count_sequences(fasta_file):
	count = 0
	for _ in SeqIO.parse(fasta_file, 'fasta'):
		count += 1
	return count