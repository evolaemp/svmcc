import argparse
import os.path
import time

from code.path_finder import get_dataset_name, find_dataset



"""
The dir where the datasets are expected to be located. If the user specifies a
dataset by name, it will be looked for in this dir.
"""
DATASETS_DIR = 'data/datasets'


"""
The directory that is expected to contain the PMI parameter files. These are
needed by the `check` and `prepare` commands.
"""
PARAMS_DIR = 'data/params'


"""
The directory where the SVM samples and targets are expected to be located.
Used by the `prepare`, `patch` and `infer` commands.
"""
VECTORS_DIR = 'data/vectors'


"""
The directory where the files containing the SVM-inferred cognate classes are
expected to be located. Used by the `infer` command.
"""
INFERRED_DIR = 'data/inferred'


"""
The directory where the `test` command looks for unit tests. It is expected to
have a `fixtures` sub-directory.
"""
TESTS_DIR = 'code/tests'



class Cli:
	"""
	Singleton that handles the user input, inits the whole machinery, and takes
	care of exiting the programme.
	"""
	
	def __init__(self):
		"""
		Constructor. Inits the argparse parser and then all the subparsers
		through the _init_* methods.
		
		Each of the latter defines a function which takes the argparse args as
		arguments and which will be called if the respective command is called.
		"""
		self.parser = argparse.ArgumentParser(description=(
			'command-line utility for manipulating the datasets'))
		
		self.subparsers = self.parser.add_subparsers(dest='command',
			title='subcommands')
		
		self._init_check()
		self._init_infer()
		self._init_prepare()
		self._init_patch()
		self._init_test()
	
	
	def _find_dataset(self, dataset):
		"""
		Returns the (path, name) tuple of the dataset identified by the given
		string (this could be the dataset's name or its path). If the dataset
		cannot be found, raises ValueError.
		
		Helper method used in most of the subcommand functions.
		"""
		if os.path.exists(dataset):
			dataset_path = dataset
		else:
			dataset_path = find_dataset(DATASETS_DIR, dataset)
			if dataset_path is None:
				raise ValueError('Could not find dataset')
		
		return dataset_path, get_dataset_name(dataset_path)
	
	
	def _init_check(self):
		"""
		Inits the subparser that handles the check command. The latter simply
		invokes check.check with the respective dataset.
		"""
		def check(args):
			from code.check import check
			
			dataset_path, _ = self._find_dataset(args.dataset)
			return check(dataset_path, PARAMS_DIR)
		
		
		usage = 'manage.py check dataset'
		description = 'dry-run the prepare command'
		
		subp = self.subparsers.add_parser('check', usage=usage,
			description=description, help=description)
		subp.add_argument('dataset', help=(
			'name of (e.g. mayan) or path to the dataset to check'))
		subp.set_defaults(func=check)
	
	
	def _init_infer(self):
		"""
		Inits the subparser that handles the infer command.
		"""
		def infer(args):
			from code.infer import infer as main
			
			start = time.time()
			
			main(VECTORS_DIR, INFERRED_DIR)
			
			end = time.time()
			return 'done in {} seconds'.format(round(end-start, 3))
		
		usage = 'manage.py infer'
		description = 'runs the svm-based automatic cognate detection'
		
		subp = self.subparsers.add_parser('infer', usage=usage,
			description=description, help=description)
		subp.set_defaults(func=infer)
	
	
	def _init_prepare(self):
		"""
		Inits the subparser that handles the prepare command.
		"""
		def prepare(args):
			from code.prepare.base import prepare, write
			
			start = time.time()
			
			dataset_path, name = self._find_dataset(args.dataset)
			
			frame = prepare(dataset_path, PARAMS_DIR)
			write(frame, name, VECTORS_DIR)
			
			end = time.time()
			return 'done in {} seconds'.format(round(end-start, 3))
		
		
		usage = 'manage.py prepare dataset'
		description = 'prepare a dataset for SVM consumption'
		
		subp = self.subparsers.add_parser('prepare', usage=usage,
			description=description, help=description)
		subp.add_argument('dataset', help=(
			'name of (e.g. mayan) or path to the dataset to prepare'))
		subp.set_defaults(func=prepare)
	
	
	def _init_patch(self):
		"""
		Inits the subparser that handles the patch command.
		"""
		def patch(args):
			from code.patch import load_samples, patch_lexstat, patch_targets
			from code.prepare.base import write_samples, write_targets
			
			start = time.time()
			
			dataset_path, dataset_name = self._find_dataset(args.dataset)
			
			if args.lexstat:
				samples = load_samples(dataset_name, OUTPUT_DIR)
				samples = patch_lexstat(dataset_path, samples)
				write_samples(samples, dataset_name, OUTPUT_DIR)
			elif args.targets:
				targets = patch_targets(dataset_path)
				write_targets(targets, dataset_name, OUTPUT_DIR)
			
			end = time.time()
			return 'patched in {} seconds'.format(round(end-start, 3))
		
		
		usage = 'manage.py patch (--lexstat | --targets) dataset'
		description = 'patch dataset samples or targets'
		
		subp = self.subparsers.add_parser('patch', usage=usage,
			description=description, help=description)
		
		subp.add_argument('dataset', help=(
			'name of (e.g. mayan) or path to the dataset to patch'))
		
		group = subp.add_mutually_exclusive_group(required=True)
		group.add_argument('--lexstat', action='store_true', help=(
			're-calculate the LexStat features only; '
			'PMI features or targets are not modified'))
		group.add_argument('--targets', action='store_true', help=(
			'prepare the targets only; samples are not modified'))
		
		subp.set_defaults(func=patch)
	
	
	def _init_test(self):
		"""
		Inits the subparser that handles the test command.
		"""
		def test(args):
			from unittest import TestLoader, TextTestRunner
			
			loader = TestLoader()
			
			if args.module:
				suite = loader.loadTestsFromName(args.module)
			else:
				suite = loader.discover(TESTS_DIR)
			
			runner = TextTestRunner(verbosity=2)
			runner.run(suite)
		
		
		usage = 'manage.py test [module]'
		description = 'run unit tests'
		
		subp = self.subparsers.add_parser('test', usage=usage,
			description=description, help=description)
		subp.add_argument('module', nargs='?', help=(
			'dotted name of the module to test; '
			'if omitted, run all tests'))
		subp.set_defaults(func=test)
	
	
	def run(self, raw_args=None):
		"""
		Parses the given arguments (if these are None, then argparse's parser
		defaults to parsing sys.argv), calls the respective subcommand function
		with the parsed arguments, and then exits.
		"""
		args = self.parser.parse_args(raw_args)
		
		if args.command is None:
			return self.parser.format_help()
		
		try:
			res = args.func(args)
		except Exception as err:
			self.parser.error(str(err))
		
		if res:
			print(res)
		
		self.parser.exit()
