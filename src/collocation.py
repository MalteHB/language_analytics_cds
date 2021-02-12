#!usr/bin/env python3

# Importing packages

import argparse

import pandas as pd
import re

from utils.utils import setting_default_data_dir, setting_default_out_dir, get_filepaths_from_data_dir, load_text



def main(args):

    print("Initiating some awesome collocation!")

    # Importing arguments from the arguments parser

    data_dir = args.data_dir

    out_dir = args.out_dir

    Collocation(data_dir=data_dir, out_dir=out_dir)

    print("DONE! Have a nice day. :-)")



class Collocation:

    def __init__(self, data_dir=None, out_dir=None, keyword="dog", window_size=5):

        self.data_dir = data_dir

        self.out_dir = out_dir

        self.keyword = keyword

        self.window_size = window_size

        if self.data_dir is None:

            self.data_dir = setting_default_data_dir()  # Setting default data directory.

        if self.out_dir is None:

            self.out_dir = setting_default_out_dir()  # Setting default output directory.

        self.out_dir.mkdir(parents=True, exist_ok=True)  # Making sure output directory exists.

        files = get_filepaths_from_data_dir(self.data_dir)  # Getting all the absolute filepaths from the data directory.

        text = load_text(files[0])

        tt = self.tokenize(text)

        p = self.get_cooccurrences(tokenized_text=tt, keyword="world", window_size=5)






        print("Done")


    def get_collocates(self, tokenized_text, keyword, window_size):

        collocates = []

        for i, word in enumerate(tokenized_text):

            if word == keyword:

                left_window = tokenized_text[i - window_size:i]

                right_window = tokenized_text[i:i + window_size + 1]

                total_window = left_window + right_window

                collocates.extend(total_window)

        unique_collocates = pd.unique(collocates)

        return unique_collocates


    def kwic(self, text, keyword, window_size=50):
        # For all regex matches
        for match in re.finditer(keyword, text):
            # first character index of match
            word_start = match.start()
            # last character index of match
            word_end = match.end()
            
            # Left window
            left_window_start = max(0, word_start-window_size)
            left_window = text[left_window_start:word_start]
            
            # Right window
            right_window_end = word_end + window_size
            right_window = text[word_end : right_window_end]
            
            # print line
            line = f"{left_window}{keyword}{right_window}"
            
            return line



    def tokenize(self, input_string):

        # Split on any non-alphanumeric character
        tokenizer = re.compile(r"\W+")

        # Tokenize
        token_list = tokenizer.split(input_string)

        # Return token_list
        return token_list

    def get_concatenated_texts(self, files):
        
        text_corpus = []

        for file in files:

            text = load_text(file)

            tokenized_text = self.tokenize(text)

            text_corpus.extend(tokenized_text)

        return text_corpus



    def get_collocation(self, keyword, tokenized_text, window_size):
        """Gets the number of occurences of a word

        Args:
            keyword (str): A string with the keyword
            tokenized_text (list): List of words
        """

        for token in tokenized_text:
            if token in keyword:
                counter += 1

        return counter



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        metavar="data_dir",
                        type=str,
                        help='A PosixPath to the data directory',
                        required=False)

    parser.add_argument('--out_dir',
                        metavar="out_dir",
                        type=str,
                        help='A path to the output directory',
                        required=False)                

    main(parser.parse_args())