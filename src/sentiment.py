#!usr/bin/env python3

"""Dictionary-based sentiment analysis with Python



Download the following CSV file from Kaggle:



https://www.kaggle.com/therohk/million-headlines



This is a dataset of over a million headlines taken from the Australian news source ABC (Start Date: 2003-02-19 ; End Date: 2020-12-31).



Calculate the sentiment score for every headline in the data. You can do this using the spaCyTextBlob approach that we covered in class or any other dictionary-based approach in Python.
Create and save a plot of sentiment over time with a 1-week rolling average
Create and save a plot of sentiment over time with a 1-month rolling average
Make sure that you have clear values on the x-axis and that you include the following: a plot title; labels for the x and y axes; and a legend for the plot
Write a short summary (no more than a paragraph) describing what the two plots show. You should mention the following points: 1) What (if any) are the general trends? 2) What (if any) inferences might you draw from them?


General instructions

For this assignment, you should upload a standalone .py script which can be executed from the command line.
Save your script as sentiment.py
Make sure to include a requirements.txt file and details about where to find the data
You can either upload the scripts here or push to GitHub and include a link - or both!
Your code should be clearly documented in a way that allows others to easily follow the structure of your script and to use them from the command line
    """
# Importing packages
import argparse

import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

from utils.utils import setting_default_data_dir, setting_default_out_dir


def main(args):

    # Importing arguments from the arguments parser

    data_dir = args.dd

    out_dir = args.od

    n_headlines = args.n_headlines

    batch_size = args.bs

    Sentiment(data_dir=data_dir, out_dir=out_dir, n_headlines=n_headlines, batch_size=batch_size)

    print("DONE! Have a nice day. :-)")


class Sentiment:
    """Calculates the sentiment scores for a given .csv file.

    Outputs plots of rolling averages across weeks and months.
    """

    def __init__(self, data_dir=None, out_dir=None, n_headlines=None, batch_size=None):

        self.data_dir = data_dir

        if self.data_dir is None:

            self.data_dir = setting_default_data_dir(assigment=3)  # Setting default data directory.

            print(f"\nData directory is not specified.\nSetting it to '{self.data_dir}'.")

        self.out_dir = out_dir

        if self.out_dir is None:

            self.out_dir = setting_default_out_dir(assignment=3)  # Setting default output directory.

            print(f"\nOutput directory is not specified.\nSetting it to '{self.out_dir}'.")

        self.out_dir.mkdir(parents=True, exist_ok=True)  # Making sure output directory exists.

        self.n_headlines = n_headlines

        self.batch_size = batch_size

        if self.batch_size is None:

            self.batch_size = 5000  # Setting batch size to default 5000
            
            print(f"\nBach size is not specified.\nSetting it to '{self.batch_size}'.")
    
        self.df = pd.read_csv(self.data_dir, nrows=self.n_headlines)  # Reading dataframe
        
        self.df_length = len(self.df)

        self.df["polarity"] = self.calculate_polarity(text_series=self.df["headline_text"], batch_size=self.batch_size)  # Calculating the polarity of each headline and adding it as a column to the pandas dataframe

        self.df = self.df.set_index(pd.to_datetime(self.df["publish_date"], format="%Y%m%d"))  # Resetting the index of the dataframe to the dates, in order to use the rolling functionality of Pandas.

        one_week_average = self.df.polarity.rolling("7d").mean()

        one_month_average = self.df.polarity.rolling("30D").mean()  # Setting a month to last approximately 30 days.

        self.save_plot(series=one_week_average,
                       title="1-week Rolling Average Polarity of News Headlines")  # ANSWER: With the weekly rolling average we are seeing a slight decrease towards '08-'09 which could be due to the financial crisis. From there, however, it just goes slightly up.

        self.save_plot(series=one_month_average,
                       title="1-month Rolling Average Polarity of News Headlines")  # ANSWER: With the monthly rolling average we are seeing the same tendency, but it seems that headlines are getting more positive towards the end of 2020. Perhaps due to the thought that COVID-19 would be a remnant of 2020. Little did they know... :(


    def calculate_polarity(self, text_series, batch_size):
        """Calculates the polarity of a text series.

        Args:
            text_series (pandas.core.series.Series): A pandas Series of texts
            batch_size (int): Size of each batch

        Returns:
            list: list of polarities for each text string
        """

        nlp = spacy.load("en_core_web_sm")

        spacy_text_blob = SpacyTextBlob()

        nlp.add_pipe(spacy_text_blob)

        polarity = [sentence._.sentiment.polarity for sentence in tqdm(nlp.pipe(text_series, batch_size=batch_size), total=len(text_series))]  # List comprehension for calculating polarity for each sentence

        return polarity


    def save_plot(self, series, title=None, filepath=None):
        """Saves a plot

        Args:
            series (pandas.core.series.Series): Series of values to plot
            title (str, optional): Title and filename of the plot. Defaults to None.
        """

        plt.plot(series, label="Rolling Average Polarity")

        plt.title(title)

        plt.legend(loc='upper right')

        plt.xlabel('Date')

        plt.ylabel('Polarity')
        
        if filepath is None:
            
            filename = title.lower().replace(' ', '_')
            
            filepath = self.out_dir / f"{filename}_n_headlines_{self.df_length}.png"

        plt.savefig(filepath)

        plt.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dd',
                        metavar="Data Directory",
                        type=str,
                        help='A PosixPath to the data directory.',
                        required=False)

    parser.add_argument('--od',
                        metavar="Output Directory",
                        type=str,
                        help='A path to the output directory.',
                        required=False)

    parser.add_argument('--n_headlines',
                        metavar="Number of headlines",
                        type=int,
                        help='The number of headlines to load from a dataframe',
                        required=False)

    parser.add_argument('--bs',
                        metavar="Batch Size",
                        type=int,
                        help='The size of each batch processed by the spaCy pipeline',
                        required=False)

    main(parser.parse_args())
