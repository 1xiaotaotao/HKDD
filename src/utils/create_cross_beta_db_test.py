import pandas as pd
import random
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import re

def create_set(table_file, clazz):
	original_aa_file = f'data/aa/Cross_Beta_DB/{clazz}.fasta'
	original_sa_file = f'data/sa/Cross_Beta_DB/{clazz}.fasta'
	output_aa_test_file = f'data/aa/Cross_Beta_DB/test_{clazz}.fasta'
	output_aa_train_file = f'data/aa/Cross_Beta_DB/train_{clazz}.fasta'    
	output_sa_train_file = f'data/sa/Cross_Beta_DB/train_{clazz}.fasta'
	df = pd.read_csv(table_file)
	seqs = df['Positive'].tolist() if clazz == 'pos' else df['Negative'].tolist()
	for i in range(len(seqs)):       
		seqs[i] = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seqs[i].upper())  # type: ignore
	aa_records = list(SeqIO.parse(original_aa_file, 'fasta'))
	sa_records = list(SeqIO.parse(original_sa_file, 'fasta'))
	aa_record = {}
	sa_record = {}
	for aa_rec, sa_rec in zip(aa_records, sa_records):
		seq_str = str(aa_rec.seq)
		aa_record[seq_str] = aa_rec
		sa_record[seq_str] = sa_rec

	test_aa = []
	idx = 0
	for test_seq in seqs:     
		if test_seq in aa_record:
			del aa_record[test_seq]
			del sa_record[test_seq]       
		record = SeqRecord(seq=Seq(test_seq), id=f'{idx:03d}_seq_{idx+1}')
		test_aa.append(record)
		idx += 1         
	SeqIO.write(test_aa, output_aa_test_file, 'fasta')
	train_seqs = random.sample(list(aa_record.keys()), 48)
	# train_seqs = list(aa_record.keys()) 
	train_aa = [aa_record[seq] for seq in train_seqs]
	train_sa = [sa_record[seq] for seq in train_seqs]
	SeqIO.write(train_aa, output_aa_train_file, 'fasta')
	SeqIO.write(train_sa, output_sa_train_file, 'fasta')
    







def create_set_for_cross_beta(table_file):
	create_set(table_file, 'pos')
	create_set(table_file, 'neg')