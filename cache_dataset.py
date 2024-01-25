"""
Code written here reads individual wiki articles and combines them as tokenized integers.
Useful for cached datasets in language modelling.
without cached dataset, articles contain mostly <pad> tokens and this extends training time.
"""

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
import os
import pickle
from tqdm import tqdm
from multiprocessing import Pool


tokenizer = Tokenizer.from_file('/URL/TO/YOUR/TOKENIZER')


tokenizer.post_processor = TemplateProcessing(
    single='<start> $A <end>',
    special_tokens=[
        ('<start>', tokenizer.token_to_id('<start>')),
        ('<end>', tokenizer.token_to_id('<end>')),
    ],
)


def tokenize_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        return tokenizer.encode(text).ids


def tokenize_file_wrapper(args):
    """
    Required for multiprocessing map functions
    """
    return tokenize_file(*args)


def tokenize_directory_multiprocess(input_dir, output_file, num_processes):
    tokenized_corpus = []
    
    _all_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if not 'combined' in file]
    file_paths = [(file_path,) for file_path in _all_files]

    with Pool(num_processes) as pool:
        tokens_list = list(tqdm(pool.imap(tokenize_file_wrapper, file_paths), total=len(file_paths)))

    for tokens in tokens_list:
        tokenized_corpus.extend(tokens)

    with open(output_file, 'wb') as output:
        pickle.dump(tokenized_corpus, output)


def tokenize_directory_singleprocess(input_dir, output_file):
    tokenized_corpus = []
    
    _all_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if not 'combined' in file]
    
    for file_path in tqdm(_all_files, total=len(_all_files)):
        tokens = tokenize_file(file_path)
        tokenized_corpus.extend(tokens)

    with open(output_file, 'wb') as output:
        pickle.dump(tokenized_corpus, output)


def test_cached_file(cached_file_path):
    if not os.path.isfile(cached_file_path):
        print(f'ERROR: cached file {cached_file_path} does not exist! Please cache the dataset first!')
        return
    
    import torch
    
    with open(cached_file_path, 'rb') as file:
        tokenized_corpus = pickle.load(file)

    tokens = torch.tensor(tokenized_corpus, dtype=torch.long)
    
    print('RAW TOKENS:')
    print(tokens)

    print()
    print(tokenizer.decode(tokens.tolist(), skip_special_tokens=False))


# REPLACE WITH YOUR OWN WIKI DUMP
corpus_directory = '../data/trwiki-20231120-pages-articles/'
output_binary_file = 'trwiki-20231120-pages-articles_cached.pkl'


if __name__ == '__main__':
    #tokenize_directory_singleprocess(corpus_directory, output_binary_file)
    tokenize_directory_multiprocess(corpus_directory, output_binary_file, os.cpu_count()-4)
    #test_cached_file(output_binary_file)

