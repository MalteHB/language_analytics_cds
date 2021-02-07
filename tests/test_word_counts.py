import os

# Import packages
import unittest
import argparse

# Import modules
from word_counts import WordCounts


class TestTrainC234(unittest.TestCase):
    """Test Train instance.

    Args:
        unittest (unittest): Test Train.
    """


    def test_train_c234(self):
        """Test analyse instance.
        """
        os.environ["DEBUG"] = 'True'

        parser = argparse.ArgumentParser()

        parser.add_argument('--data_dir',
                        metavar="data_dir",
                        type=str,
                        help='A PosixPath to the data directory',
                        required=False)

        # PIPELINE #
        word_counts.main(args)

        self.assertTrue(os.path.exists("unittest_data"))

# if __name__ == '__main__':
#     unittest.main()
