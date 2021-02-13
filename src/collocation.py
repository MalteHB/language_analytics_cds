#!usr/bin/env python3

# Importing packages

import argparse
import re

import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.utils import setting_default_data_dir, setting_default_out_dir, get_filepaths_from_data_dir, load_text


def main(args):

    print("Initiating some awesome collocation!")

    # Importing arguments from the arguments parser

    data_dir = args.data_dir

    out_dir = args.out_dir

    Collocation(data_dir=data_dir, out_dir=out_dir)

    print("DONE! Have a nice day. :-)")



class Collocation:

    def __init__(self, data_dir=None, out_dir=None, keyword="world", window_size=2, number_of_files=10):

        self.data_dir = data_dir

        self.out_dir = out_dir

        self.keyword = keyword

        self.window_size = window_size

        self.number_of_files = number_of_files

        if self.data_dir is None:

            self.data_dir = setting_default_data_dir()  # Setting default data directory.

        if self.out_dir is None:

            self.out_dir = setting_default_out_dir()  # Setting default output directory.

        self.out_dir.mkdir(parents=True, exist_ok=True)  # Making sure output directory exists.

        files = get_filepaths_from_data_dir(self.data_dir)  # Getting all the absolute filepaths from the data directory.

        print(f"Initiating collocation calculations for the files in '{self.data_dir}'")

        for file in tqdm(files[0:self.number_of_files]):  # tqdm for progress bar and limiting the amount of files, because these calculations take alot of time.

            tokenized_text = self.get_tokenized_text(file)

            collocates = self.word_collocates(tokenized_text=tokenized_text, keyword=keyword, window_size=window_size)

            raw_frequencies = []

            MIs = []

            # print(f"Calculating collocates for '{file.name}'.")

            for collocate in tqdm(collocates):  # tqdm for progress bar

                raw_frequency = self.raw_frequency(tokenized_text=tokenized_text, keyword=collocate)  # Raw frequency of collocate

                O11 = self.joint_frequency(tokenized_text=tokenized_text, keyword=keyword, collocate=collocate, window_size=window_size) # Joint frequency of keyword and collocate

                O12 = self.disjoint_frequency(tokenized_text=tokenized_text, keyword=keyword, collocate=collocate, window_size=window_size)  # Disjoint frequency of keyword and collocate

                O21 = self.disjoint_frequency(tokenized_text=tokenized_text, keyword=collocate, collocate=keyword, window_size=window_size)  # Disjoint frequency of collocate as keyword and keyword as collocate

                O22 = self.n_words_without_keyword_and_collocate(tokenized_text=tokenized_text, keyword=keyword, collocate=collocate)  # All the words in the corpus that are not either the keyword nor the collocate

                N = O11 + O12 + O21 + O22  # There is doubt whether to calculate N like this or by simply taking len(tokenized_text)

                R1 = O11 + O12  # Calculating R1

                C1 = O11 + O21  # Calculating C1

                # Expected
                E11 = (R1 * C1 / N)  # Expected frequency

                # return MI
                MI = np.log2(O11 / E11)  # Mutual information

                raw_frequencies.append(raw_frequency)

                MIs.append(MI)

            data_dict = {"collocate": collocates,
                         "raw_frequency": raw_frequencies,
                         "MI": MIs}

            df = pd.DataFrame(data=data_dict)

            df = df.sort_values("raw_frequency", ascending=False)  # Sorting collocates with highest frequency at the top.

            write_path = self.out_dir / f"collocates_{file.stem}.csv"
            
            df.to_csv(write_path)

        print("Done")


    def get_tokenized_text(self, file):
        
        text = load_text(file)

        tokenized_text = self.tokenize(text)

        return tokenized_text


    def word_collocates(self, tokenized_text, keyword, window_size):

        collocates = []

        for i, word in enumerate(tokenized_text):

            if word == keyword:

                left_window = tokenized_text[max(0, i - window_size):i]

                right_window = tokenized_text[i:(i + window_size + 1)]

                total_window = left_window + right_window

                collocates.extend(total_window)

        unique_collocates = pd.unique(collocates)

        return unique_collocates


    def raw_frequency(self, tokenized_text, keyword):

        word_counter = 0

        for word in tokenized_text:

            if word == keyword:

                word_counter += 1

        return word_counter


    def joint_frequency(self, tokenized_text, keyword, collocate, window_size):

        joint_frequency = 0

        for i, word in enumerate(tokenized_text):

            if word == keyword:

                left_window = tokenized_text[max(0, i - window_size):i]

                right_window = tokenized_text[i:(i + window_size + 1)]

                total_window = left_window + right_window

                if keyword and collocate in total_window:

                    joint_frequency += 1

        return joint_frequency


    def disjoint_frequency(self, tokenized_text, keyword, collocate, window_size):

        disjoint_frequency = 0

        for i, word in enumerate(tokenized_text):

            if word == keyword:

                left_window = tokenized_text[max(0, i - window_size):i]

                right_window = tokenized_text[i:(i + window_size + 1)]

                total_window = left_window + right_window

                if keyword in total_window and collocate not in total_window:

                    disjoint_frequency += 1

        return disjoint_frequency


    def n_words_without_keyword_and_collocate(self, tokenized_text, keyword, collocate):

        word_counter = 0

        for word in tokenized_text:

            if word not in keyword and word not in collocate:

                word_counter += 1

        return word_counter


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