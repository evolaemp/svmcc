import csv
import os.path

from code.prepare.base import load_data, load_targets
from code.prepare.lexstat import set_schema, make_wordlist, calc_lexstat
from code.prepare.pmi import get_pairs
from code.prepare.utils import is_asjp_data



def patch_lexstat(dataset_path, samples):
	"""
	Re-calculates the dataset's LexStat scores.
	Receives and returns {sample_id: [feature,]}.
	"""
	data = load_data(dataset_path)
	lang_pairs = [(a, b) for a in data.keys() for b in data.keys() if a < b]
	
	with set_schema('asjp' if is_asjp_data(data) else 'ipa'):
		lingpy_wordlist = make_wordlist(data, dataset_path)
		
		for lang1, lang2 in lang_pairs:
			scores = calc_lexstat(lang1, lang2, lingpy_wordlist)
			for key, score in scores.items():
				assert key in samples
				samples[key].extend(list(score))
	
	return samples



def load_samples(dataset_name, output_dir):
	"""
	Returns {sample_id: [feature,]} for the named dataset.
	"""
	samples_dir = os.path.join(output_dir, 'samples')
	assert os.path.exists(samples_dir)
	
	file_path = os.path.join(samples_dir, dataset_name +'.tsv')
	assert os.path.exists(file_path)
	
	samples = {}
	
	with open(file_path) as f:
		reader = csv.reader(f, delimiter='\t')
		next(reader)
		
		for line in reader:
			samples[line[0]] = line[1:7]
			assert len(samples[line[0]]) == 6
	
	return samples



def patch_targets(dataset_path):
	"""
	Re-outputs the dataset's targets.
	"""
	data = load_data(dataset_path)
	sample_keys = []
	
	lang_pairs = [(a, b) for a in data.keys() for b in data.keys() if a < b]
	for lang1, lang2 in lang_pairs:
		syn, _ = get_pairs(lang1, lang2, data)
		sample_keys.extend(list(syn.keys()))
	
	return load_targets(dataset_path, sample_keys, data.keys())
