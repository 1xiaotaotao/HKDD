import random
from Bio import SeqIO



def s(clazz):
	original_aa_file = f'data/aa/Cross_Beta_DB/{clazz}.fasta'
	original_sa_file = f'data/sa/Cross_Beta_DB/{clazz}.fasta'
	output_aa_file = f'data/aa/Cross_Beta_DB/new_{clazz}.fasta'
	output_sa_file = f'data/sa/Cross_Beta_DB/new_{clazz}.fasta'
	aa_records = list(SeqIO.parse(original_aa_file, 'fasta'))
	sa_records = list(SeqIO.parse(original_sa_file, 'fasta'))
	aa_record = {}
	sa_record = {}
	for aa_rec, sa_rec in zip(aa_records, sa_records):
		seq_str = str(aa_rec.seq)
		aa_record[seq_str] = aa_rec
		sa_record[seq_str] = sa_rec

	train_seqs = random.sample(list(aa_record.keys()), 61)
	train_aa = [aa_record[seq] for seq in train_seqs]
	train_sa = [sa_record[seq] for seq in train_seqs]
	SeqIO.write(train_aa, output_aa_file, 'fasta')
	SeqIO.write(train_sa, output_sa_file, 'fasta')


def select():
	s('neg')