import csv
import os.path

from code.prepare.base import load_data, load_targets
from code.prepare.lexstat import set_schema, make_wordlist, calc_lexstat
from code.prepare.pmi import get_pairs
from code.prepare.utils import explode_sample_id, is_asjp_data



"""
The columns of a vectors file
"""
VECTORS_COLS = ['gloss', 'l1', 'w1', 'cc1', 'l2', 'w2', 'cc2', 'feature1',
		'feature2', 'feature3', 'feature4', 'feature5', 'feature6',
		'lexstat_simAA', 'lexstat_simBB', 'lexstat_simAB', 'feature7',
		'target', 'db']



def patch_lexstat(dataset_path, vectors_path):
	"""
	Re-calculates the dataset's LexStat scores.
	"""
	data = load_data(dataset_path)
	
	all_langs = set(data.keys())
	lang_pairs = [(a, b) for a in all_langs for b in all_langs if a < b]
	
	lexstat_samples = {}
	
	with set_schema('asjp' if is_asjp_data(data) else 'ipa'):
		lingpy_wordlist = make_wordlist(data, dataset_path)
		
		for lang1, lang2 in lang_pairs:
			scores = calc_lexstat(lang1, lang2, lingpy_wordlist)
			for key, score in scores.items():
				key = explode_sample_id(key, all_langs)
				lexstat_samples[key] = list(score)
	
	gloss_d = {}  # gloss to global gloss id
	with open(dataset_path) as f:
		reader = csv.DictReader(f, delimiter='\t')
		for row in reader:
			gloss_d[row['gloss']] = row['global_id']
	
	vectors = []
	with open(vectors_path, newline='', encoding='utf-8') as f:
		vectors = [row for row in csv.DictReader(f)]
	
	for vector in vectors:
		subkey = (gloss_d[vector['gloss']], vector['l1'], vector['l2'])
		pots = [key for key in lexstat_samples.keys() if key[:3] == subkey]
		scores = lexstat_samples.pop(pots[0])
		vector['lexstat_simAA'] = '{:.10f}'.format(scores[0])
		vector['lexstat_simBB'] = '{:.10f}'.format(scores[1])
		vector['lexstat_simAB'] = '{:.10f}'.format(scores[2])
	
	assert len(lexstat_samples) == 0
	
	with open(vectors_path, 'w', newline='', encoding='utf-8') as f:
		writer = csv.DictWriter(f, VECTORS_COLS,
				lineterminator='\n')  # be consistent with pandas output
		writer.writeheader()
		for vector in vectors:
			writer.writerow(vector)



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
