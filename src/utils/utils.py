#!usr/bin/env python3

from pathlib import Path


def setting_default_data_dir(assigment=4):
    """Setting a default data directory

    Returns:
        PosixPath: Data directory
    """

    if assigment == 2:
    
        root_dir = Path.cwd()  # Setting root directory.

        data_dir = root_dir / "data" / "100_english_novels" / "corpus"  # Setting data directory.

    if assigment == 3:

        root_dir = Path.cwd()  # Setting root directory.

        data_dir = root_dir / "data" / "abcnews-date-text.csv"  # Setting data directory.

    if assigment == 4:

        root_dir = Path.cwd()

        data_dir = root_dir / "data" / "edges_df.csv"

    return data_dir


def setting_default_out_dir(assignment=5):
    """Setting a default Output directory

    Returns:
        PosixPath: Output directory
    """

    if assignment in (1, 2, 3, 6):
    
        root_dir = Path.cwd()  # Setting root directory.

        out_dir = root_dir / "out"  # Setting data directory.

        return out_dir

    if assignment == 4:

        root_dir = Path.cwd()

        graph_out_dir = root_dir / "out" / "viz"

        data_out_dir = root_dir / "out"

        return graph_out_dir, data_out_dir

    if assignment == 5:

        root_dir = Path.cwd()

        model_out_dir = root_dir / "out" / "models"

        return model_out_dir


def get_filepaths_from_data_dir(data_dir, file_extension="*.txt"):
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

    filename = file.name  # I take the last snippet of the path which is the file and the file extension.

    return filename


def load_text(file):
    """Loads an image.

    Args:
        file (PosixPath): A path to an image file.

    Returns:
        numpy.ndarray: NumPy Array containg all the pixels for the image.
    """

    # Read each file.

    with open(file, encoding="utf-8") as f:

        try:

            text = f.read()

        except TypeError:

            print("wtf")

        f.close()

    return text
