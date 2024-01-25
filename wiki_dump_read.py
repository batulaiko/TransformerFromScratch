"""
This code reads the wikipedia dump xml file and preprocesses the text.
"""

from wiki_dump_reader import Cleaner, iterate
import re
from tqdm import tqdm
import os
from langdetect import detect


cleaner = Cleaner()

# If False, runs write_seperate
RUN_COMBINED = False

# REPLACE WITH YOUR OWN WIKI DUMP
# from: https://dumps.wikimedia.org/trwiki/20231120/
PATTERN = r'\b\w+\b' #r'\b\w+\b|\s'
DUMP_FILE = '../data/trwiki-20231120-pages-articles.xml'
SAVE_DIR = '../data/trwiki-20231120-pages-articles'
COMBINED_FILE_NAME = 'combined.txt'


os.makedirs(SAVE_DIR, exist_ok=True)


def extrach_matches(text, pattern):
    matches = re.findall(pattern, text)
    return ' '.join(matches)


average = lambda x: sum(x)/len(x)


hyperparameters = {
    'language': 'tr',
    'min_total_len': 50,
    'min_line_word_len': 9,
    'min_words': 1
}


stats = {
    'char_lens': [],
    'word_lens': [],
    'line_lens': [],
    'line_word_len': []
}


def remove_file(root_dir, filename):
    if os.path.isfile(os.path.join(root_dir, filename)):
        os.remove(os.path.join(root_dir, filename))
        print(os.path.join(root_dir, filename), 'is removed!')


def write_seperate(dump_file, hyperparameters, stats):
    i = 0
    skipped = 0

    for title, text in tqdm(iterate(dump_file)):
    
        text = cleaner.clean_text(text)
        cleaned_text, links = cleaner.build_links(text)
        
        ###### TOTAL LEN ######
        if len(cleaned_text) < hyperparameters['min_total_len']:
            #print(f'[<--->] {i} is skipped')
            skipped += 1
            continue
        #######################

        ### LANGUAGE DETECT ###
        detected_lang = detect(cleaned_text)
    
        if detected_lang != 'tr':
            skipped += 1
            continue
        #######################
    
        #print(f'#### {i} ####')
        
        #print(text)
        #print(cleaned_text)
        #print(extrach_matches(cleaned_text, PATTERN))
    
        _write_str = ''
    
        for line in cleaned_text.splitlines():
            #print(f'[{len(line.split())}]  ', line)
            stats['line_word_len'].append(len(line.split()))
    
            if len(line.split()) > hyperparameters['min_line_word_len']:
                _write_str += f'{line} \n'
    
        # Make sure that _write_str is not empty!
        if len(re.findall(PATTERN, _write_str)) > hyperparameters['min_words']:
            with open(os.path.join(SAVE_DIR, f'article_{i}.txt'), 'w') as f:
                f.write(_write_str)
    
        #print(links)
    
        """
        print(f'TOTAL LEN: {len(cleaned_text)}')
        print(f'TOTAL WORDS: {len(cleaned_text.split())}')
        print(f'TOTAL LINES: {len(cleaned_text.splitlines())}')
        print('*'*50)
        print('\n\n')
        """
    
        stats['char_lens'].append(len(cleaned_text))
        stats['word_lens'].append(len(cleaned_text.split()))
        stats['line_lens'].append(len(cleaned_text.splitlines()))
    
        i += 1
    
        """
        if i > 20:
            break
        """
        
    return i, skipped


def write_combined(dump_file, hyperparameters):
    i = 0
    skipped = 0

    output_file = open(os.path.join(SAVE_DIR, 'combined.txt'), 'a')
    
    for title, text in tqdm(iterate(dump_file)):
    
        text = cleaner.clean_text(text)
        cleaned_text, links = cleaner.build_links(text)
    
        ###### TOTAL LEN ######
        if len(cleaned_text) < hyperparameters['min_total_len']:
            #print(f'[<--->] {i} is skipped')
            skipped += 1
            continue
        #######################

        ### LANGUAGE DETECT ###
        detected_lang = detect(cleaned_text)
    
        if detected_lang != 'tr':
            skipped += 1
            continue
        #######################

        _write_str = "<start> "
        
        for line in cleaned_text.splitlines():
            #print(f'[{len(line.split())}]  ', line)
            stats['line_word_len'].append(len(line.split()))
    
            if len(line.split()) > hyperparameters['min_line_word_len']:
                _write_str += f'{line}'

        _write_str += " <end>\n"

        if len(re.findall(PATTERN, _write_str)) > hyperparameters['min_words']:
            output_file.write(_write_str)
    
        i += 1

    output_file.close()
    
    return i, skipped


if __name__ == '__main__':
    if RUN_COMBINED:
        remove_file(SAVE_DIR, COMBINED_FILE_NAME)

    if RUN_COMBINED:
        i, skipped = write_combined(DUMP_FILE, hyperparameters)
    else:
        i, skipped = write_seperate(DUMP_FILE, hyperparameters, stats)

    print(f'Total: {i} articles processed')
    print(f'Total: {skipped} articles skipped')

    if not RUN_COMBINED:
        for k,v in stats.items():
            print(f'Stat: {k}, Average: {round(average(v))}')
    
    print('All done!')
