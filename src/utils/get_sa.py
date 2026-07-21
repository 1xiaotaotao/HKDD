import os
import glob
from pathlib import Path

from tqdm import tqdm

from SaProt.utils.foldseek_util import get_struc_seq
import logging

logger = logging.getLogger()


def get_sa(input_dir, output_file, foldseek_path, plddt_mask=True):
	cif_files = glob.glob(os.path.join(input_dir, '*.cif'))
	cif_files.sort()
	with open(output_file, 'w') as f:
		for cif_path in tqdm(cif_files):
			file_name = Path(cif_path).stem
			parsed_seqs = get_struc_seq(
				foldseek_path,
				cif_path,
				['A'],
				plddt_mask=plddt_mask
			)['A']
			_, _, combined_seq = parsed_seqs
			f.write(f'>{file_name}\n')
			f.write(f'{combined_seq}\n')
	
	logger.info(f'save: {output_file}')

