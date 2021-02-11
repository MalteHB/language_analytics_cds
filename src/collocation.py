#!usr/bin/env python3

# Importing packages
import os

from pathlib import Path
import argparse

import pandas as pd
import re

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

            self.data_dir = self.setting_default_data_dir()  # Setting default data directory.

        if self.out_dir is None:

            self.out_dir = self.setting_default_out_dir()  # Setting default output directory.

        self.out_dir.mkdir(parents=True, exist_ok=True)  # Making sure output directory exists.

        files = self.get_filepaths_from_data_dir(self.data_dir)  # Getting all the absolute filepaths from the data directory.

        for file in files:

            out_path = self.out_dir / file.name

            text = self.load_text(file)

            tokenized_text = self.tokenize(text)

            keyword_occurences = self.get_word_occurence(keyword, tokenized_text)

            print("Done")

            

    def setting_default_data_dir(self):
        """Setting a default data directory

        Returns:
            PosixPath: Data directory
        """

        root_dir = Path.cwd()  # Setting root directory.

        data_dir = root_dir / "data" / "cenlab" / "texts"  # Setting data directory.

        return data_dir

    def setting_default_out_dir(self):
        """Setting a default Output directory

        Returns:
            PosixPath: Output directory
        """
        root_dir = Path.cwd()  # Setting root directory.

        data_dir = root_dir / "out"  # Setting data directory.

        return data_dir

    def get_filepaths_from_data_dir(self, data_dir, file_extension="*.txt"):
        """Creates a list containing paths to filenames in a data directoryl

        Args:
            data_dir (PosixPath): PosixPath to the data directory.
            file_extension (str): A string with the given file extension you want to extract.
        """

        files = [file for file in data_dir.glob(file_extension) if file.is_file()]  # Using list comprehension to get all the file names if they are files.

        return files

    def get_filename(self, file):
        """Creates a list of filenames in a directory.

        Args:
            files (list): List of file paths

        Returns:
            filename: list of filenames
        """

        filename = os.path.split(file)[-1]  # I take the last snippet of the path which is the file and the file extension.

        return filename

    def load_text(self, file):
        """Loads an image.

        Args:
            file (PosixPath): A path to an image file.

        Returns:
            numpy.ndarray: NumPy Array containg all the pixels for the image.
        """

        # Read each file.

        with open(file, encoding="utf-8") as f:

            text = f.read()

            f.close()

        return text

   
    def tokenize(self, input_string):
        
        # Split on any non-alphanumeric character
        tokenizer = re.compile(r"\W+")
        
        # Tokenize
        token_list = tokenizer.split(input_string)
        
        # Return token_list
        return token_list

    def get_word_occurence(self, keyword, tokenized_text):
        """Gets the number of occurences of a word

        Args:
            keyword (str): A string with the keyword
            tokenized_text (list): List of words
        """

        counter = 0

        for token in tokenized_text:
            if token in keyword:
                counter += 1

        return counter
        # If keyword is in text, increase the counter.


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