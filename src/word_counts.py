# Importing packages
import os

from pathlib import Path
import pandas as pd
import argparse


def main(args):

    data_dir = args.data_dir

    WordCounts(data_dir)


class WordCounts:

    def __init__(self, data_dir=None):

        self.data_dir = data_dir

        if data_dir is None:

            data_dir = self.setting_default_data_dir()

        files = self.get_filepaths_from_data_dir(data_dir)

        filenames = self.get_filenames(files)

        total_words = self.get_total_words(files)

        unique_words = self.get_total_unique_words(files)

        df = self.get_pandas_dataframe(filenames=filenames,
                                       total_words=total_words,
                                       unique_words=unique_words)

        dataframe_path = "word_counts.csv"

        df.to_csv(dataframe_path)  # Writing data to csv.

        print(f"DONE! Created the file: '{dataframe_path}'. Have a nice day. :-)")

    def setting_default_data_dir(self):

        root_dir = Path.cwd()  # Setting root directory.

        data_dir = root_dir / "data" / "100_english_novels" / "corpus"  # Setting data directory.

        return data_dir

    def get_filepaths_from_data_dir(self, data_dir):
        """Creates a list containing paths to filenames in a data directoryl

        Args:
            data_dir (PosixPath): PosixPath to the data directory.
        """

        files = [file for file in data_dir.glob("*.txt") if file.is_file()]  # Using list comprehension to get all the file names if they are files.

        return files

    def get_filenames(self, files):
        """Creates a list of filenames in a directory.

        Args:
            files (list): List of file paths

        Returns:
            filename: list of filenames
        """

        filename = []  # Creating empty list

        # Loop for iterating through the different files.
        for file in files:

            novel_file_name = os.path.split(file)[-1]  # I take the last snippet of the path, which is the novel filename.

            filename.append(novel_file_name)  # Append each filename to the list.

        return filename


    def get_total_words(self, files):
        """Gets the total number of words from all the files

        Args:
            files (list): List of ".txt" file paths

        Returns:
            [list]: List of total words per file in the input list
        """

        total_words = []  # Creating empty list

        # Loop for iterating through the different files.
        for file in files:

            # Read each file.
            with open(file, encoding="utf-8") as f:

                novel = f.read()

                f.close()

            tokens = novel.split()  # I split the tokens by whitespace. Do note that to get a proper tokenization, more clever solutions could be applied. See fx NLTK's tokenizer.

            total_words.append(len(tokens))  # I append the length of the tokens from each novel.

        return total_words

    def get_total_unique_words(self, files):

        unique_words = []  # Creating empty list

        # Loop for iterating through the different files.
        for file in files:

            # Read each file.
            with open(file, encoding="utf-8") as f:

                novel = f.read()

                f.close()

            tokens = novel.split()  # I split the tokens by whitespace. Do note that to get a proper tokenization, more clever solutions could be applied. See fx NLTK's tokenizer.

            unique_tokens = set(tokens)  # Create a set of unique tokens.

            unique_words.append(len(unique_tokens))  # I append the number of unique tokens from each novel.

        return unique_words

    def get_pandas_dataframe(self, filenames, total_words, unique_words):

        # Creating Python dictionary to use for pandas. If there are more efficient ways to create a pandas DataFrame from scratch - please let me know :-)
        data_dict = {"filename": filenames,
                     "total_words": total_words,
                     "unique_words": unique_words}

        df = pd.DataFrame(data=data_dict)

        return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        metavar="data_dir",
                        type=str,
                        help='A PosixPath to the data directory',
                        required=False)

    main(parser.parse_args())
