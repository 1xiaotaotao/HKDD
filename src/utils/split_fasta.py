from Bio import SeqIO
import random
import logging
logger = logging.getLogger()


def split_fasta_8_2(dataset):
	aa_pos = list(SeqIO.parse(f'data/aa/{dataset}/pos.fasta', 'fasta'))
	sa_pos = list(SeqIO.parse(f'data/sa/{dataset}/pos.fasta', 'fasta'))
	aa_neg = list(SeqIO.parse(f'data/aa/{dataset}/neg.fasta', 'fasta'))
	sa_neg = list(SeqIO.parse(f'data/sa/{dataset}/neg.fasta', 'fasta'))
	
	assert len(aa_pos) == len(sa_pos), 'aa sequences and sa sequences should have same length'
	assert len(aa_neg) == len(sa_neg), 'aa sequences and sa sequences should have same length'
	
	data = {
		'pos': list(zip(aa_pos, sa_pos)),
		'neg': list(zip(aa_neg, sa_neg))
	}
	
	for label in ['pos', 'neg']:
		random.shuffle(data[label])
		total = len(data[label])
		split_point = int(total * 0.8)
		
		train = data[label][:split_point]
		test = data[label][split_point:]
		
		aa_dir = f'data/aa/{dataset}'
		SeqIO.write([p[0] for p in train], f'{aa_dir}/train_{label}.fasta', 'fasta')
		SeqIO.write([p[0] for p in test], f'{aa_dir}/test_{label}.fasta', 'fasta')
		
		sa_dir = f'data/sa/{dataset}'
		SeqIO.write([p[1] for p in train], f'{sa_dir}/train_{label}.fasta', 'fasta')
		SeqIO.write([p[1] for p in test], f'{sa_dir}/test_{label}.fasta', 'fasta')
	logger.info('all files have split to train and test')
