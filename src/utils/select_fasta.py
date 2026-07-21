from Bio import SeqIO
import random


def random_select_fasta(dataset, ratio, mode):
	aa_pos_file = f'data/aa/{dataset}/{mode}_pos.fasta'
	aa_neg_file = f'data/aa/{dataset}/{mode}_neg.fasta'
	sa_pos_file = f'data/sa/{dataset}/{mode}_pos.fasta'
	sa_neg_file = f'data/sa/{dataset}/{mode}_neg.fasta'
	aa_pos_records = list(SeqIO.parse(aa_pos_file, 'fasta'))
	aa_neg_records = list(SeqIO.parse(aa_neg_file, 'fasta'))
	sa_pos_records = list(SeqIO.parse(sa_pos_file, 'fasta'))
	sa_neg_records = list(SeqIO.parse(sa_neg_file, 'fasta'))
	
	assert len(aa_pos_records) == len(sa_pos_records) and len(aa_neg_records) == len(sa_neg_records)
	
	for (aa_rec, sa_rec), file_prefix in [((aa_pos_records, sa_pos_records), 'pos'),
	                                      ((aa_neg_records, sa_neg_records), 'neg')]:
		paired = list(zip(aa_rec, sa_rec))
		random.shuffle(paired)
		selected = paired[:int(len(paired) * ratio)]
		
		SeqIO.write([p[0] for p in selected], f'data/aa/{dataset}/{mode}_{file_prefix}.fasta', 'fasta')
		SeqIO.write([p[1] for p in selected], f'data/sa/{dataset}/{mode}_{file_prefix}.fasta', 'fasta')

def select_fasta(dataset, ratio=0.5):
	random_select_fasta(dataset, ratio, 'train')
	# random_select_fasta(dataset, ratio, 'test')