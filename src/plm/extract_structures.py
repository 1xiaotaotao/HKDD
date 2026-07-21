from src.utils.read_fasta_file import read_fasta

import esm
import torch
from tqdm import tqdm
import logging

from Bio.PDB import PDBParser, MMCIFIO
from io import StringIO
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
class ExtractStructures:
	def __init__(self, input_pos_file, input_neg_file, out_pos_dir, out_neg_dir):
		self.input_pos_file = input_pos_file
		self.input_neg_file = input_neg_file
		self.out_pos_dir = out_pos_dir
		self.out_neg_dir = out_neg_dir
		self.model = None
		self.logger = logging.getLogger(self.__class__.__name__)
	
	
	def extract_process(self, seq_ids, sequences, out_dir):
		self.logger.info(f'Extracting structures for sequences')
		for idx, (seq_id, sequence) in enumerate(tqdm(zip(seq_ids, sequences), total=len(sequences))):
			safe_id_base = seq_id.replace('/', '_')
			safe_id = f'{idx:03d}_{safe_id_base}'
			cif_path = f'{out_dir}/{safe_id}.cif'
			with torch.no_grad():
				pdb = self.model.infer_pdb(sequence)
			parser = PDBParser(QUIET=True)
			structure = parser.get_structure('protein', StringIO(pdb))
			
			io_cif = MMCIFIO()
			io_cif.set_structure(structure)
			io_cif.save(str(cif_path))
		self.logger.info(f'structure file have saved to {out_dir}')
	
	def extract(self):
		self.logger.info('extracting structures by EsmFold')
		model = esm.pretrained.esmfold_v1()
		self.model = model.eval().cuda()
		self.model.set_chunk_size(128)
		
		pos_ids, pos_sequences = read_fasta(self.input_pos_file)
		neg_ids, neg_sequences = read_fasta(self.input_neg_file)
		
		self.logger.info('extracting structures of pos by esm_fold_v1')
		self.extract_process(pos_ids, pos_sequences, self.out_pos_dir)
		self.logger.info('extracting structures of neg by esm_fold_v1')
		self.extract_process(neg_ids, neg_sequences, self.out_neg_dir)
		



