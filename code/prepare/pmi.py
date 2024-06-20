import math


from code.prepare.utils import make_sample_id
from code.prepare.utils import is_asjp_data, ipa_to_asjp, asjp_to_asjp



def get_pairs(lang1, lang2, data):
	"""
	Returns the pairs of synonymous (i.e. having the same Concepticon ID) and
	non-synonymous words.
	The synonymous pairs are {pair_id: (ipa1, ipa2)}.
	The non-synonymous pairs are [(ipa1, ipa2),].
	"""
	syn = {}
	non_syn = []
	
	li1 = [(g, i, t) for g in data[lang1] for i, t in enumerate(data[lang1][g])]
	li2 = [(g, i, t) for g in data[lang2] for i, t in enumerate(data[lang2][g])]
	
	for gloss1, index1, ipa1 in li1:
		for gloss2, index2, ipa2 in li2:
			if gloss1 == gloss2:
				id_ = make_sample_id(gloss1, lang1, lang2, index1+1, index2+1)
				syn[id_] = (ipa1, ipa2)
			else:
				non_syn.append((ipa1, ipa2))
	
	return syn, non_syn



def get_asjp_data(data, params):
	"""
	Returns copy of the given {lang: {gloss: [ipa,]}}, IPA replaced with ASJP.
	"""
	if is_asjp_data(data):
		func = asjp_to_asjp
	else:
		func = ipa_to_asjp
	
	asjp_data = dict.fromkeys(data.keys(), {})
	for lang in asjp_data:
		asjp_data[lang] = {
			gloss: [func(ipa, params) for ipa in data[lang][gloss]]
			for gloss in data[lang]
		}
	
	return asjp_data


#%%
def calc_pmi(string1, string2, params):
    """
    Compute the Needleman-Wunsch alignment score between string1 and string2.
    Expects params dict to contain the keys: logodds, gap_penalties.
    """
    logodds = params['logodds']
    gap_open, gap_extend = params['gap_penalties']
    
    len1, len2 = len(string1), len(string2)
    
    # Initialize the scoring matrix
    score_matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    gap_matrix_1 = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    gap_matrix_2 = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    # Initialize gap penalties
    for i in range(1, len1 + 1):
        score_matrix[i][0] = gap_open + (i - 1) * gap_extend
        gap_matrix_1[i][0] = gap_open + (i - 1) * gap_extend
        gap_matrix_2[i][0] = float('-inf')
        
    for j in range(1, len2 + 1):
        score_matrix[0][j] = gap_open + (j - 1) * gap_extend
        gap_matrix_1[0][j] = float('-inf')
        gap_matrix_2[0][j] = gap_open + (j - 1) * gap_extend

    # Fill the scoring matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            match_score = logodds.get((string1[i - 1], string2[j - 1]), 0)
            gap_matrix_1[i][j] = max(score_matrix[i - 1][j] + gap_open, gap_matrix_1[i - 1][j] + gap_extend)
            gap_matrix_2[i][j] = max(score_matrix[i][j - 1] + gap_open, gap_matrix_2[i][j - 1] + gap_extend)
            score_matrix[i][j] = max(score_matrix[i - 1][j - 1] + match_score, gap_matrix_1[i][j], gap_matrix_2[i][j])

    # The alignment score is in the bottom-right cell
    return score_matrix[len1][len2]

#%%



def prepare_lang_pair(lang1, lang2, data, params):
	"""
	The transcriptions of the data {} must be ASJP.
	The output is {pair_id: [feature,]}.
	"""
	syn, non_syn = get_pairs(lang1, lang2, data)
	
	pmi = {key: calc_pmi(p[0], p[1], params) for key, p in syn.items()}
	
	non_syn_pmi = [calc_pmi(p[0], p[1], params) for p in non_syn]
	div = len(non_syn) + 1
	calib_pmi = {key: (sum([1 for i in non_syn_pmi if i>pmi[key]])+1)/div for key in syn}
	
	feature3 = {key: -math.log(value) for key, value in calib_pmi.items()}
	feature4 = sum(feature3.values()) / len(feature3)
	feature5 = math.log(feature4)
	
	samples = {}
	for key in syn.keys():
		samples[key] = [
			pmi[key], calib_pmi[key], feature3[key], feature4, feature5
		]
	
	return samples
