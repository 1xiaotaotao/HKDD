import logging
from Bio import SeqIO
logger = logging.getLogger(__name__)


def read_sequences(fasta_file):
	logging.info(f'reading fasta file {fasta_file}')
	sequences = []
	with open(fasta_file, 'r') as f:
		for record in SeqIO.parse(f, 'fasta'):
			sequences.append(str(record.seq))
	return sequences

def read_sequences_truncated(fasta_file):
	logging.info(f'reading fasta file {fasta_file}')
	sequences = []
	with open(fasta_file, 'r') as f:
		for record in SeqIO.parse(f, 'fasta'):
			sequences.append(str(record.seq[:510]))
	return sequences

def read_fasta(fasta_file):
	logging.info(f'reading fasta file {fasta_file}')
	seq_ids = []
	sequences = []
	with open(fasta_file, 'r') as f:
		for record in SeqIO.parse(f, 'fasta'):
			seq_ids.append(record.id)
			sequences.append(str(record.seq)[:510])
	return seq_ids, sequences