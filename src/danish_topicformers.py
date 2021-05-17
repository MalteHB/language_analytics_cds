#!usr/bin/env python3
# Importing packages
import argparse
import numpy as np
from bertopic import BERTopic
from datasets import load_dataset

import re
import emoji
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import warnings

from utils.utils import setting_default_out_dir


def main(args):

    # Importing arguments from the arguments parser

    dataset = args.ds
    
    embedding_model = args.em

    topic_number = args.tn

    out_dir = args.od

    save_model = args.sm

    save_path = args.sp

    load_model = args.lm

    load_path = args.lp

    keep_hashtag_text = args.keep_hashtag_text

    remove_stopwords = args.remove_stopwords

    danish_topic_model = DanishTopicFormers(dataset=dataset,
                                            embedding_model=embedding_model,
                                            out_dir=out_dir)

    if load_model:
        
        danish_topic_model.load_dataset_and_preprocess_data(keep_hashtag_text=keep_hashtag_text,
                                                            remove_stopwords=remove_stopwords)
        
        danish_topic_model.create_topic_model()

        danish_topic_model.load_model(path=load_path)
        
        danish_topic_model.train_topic_model()

    else:

        danish_topic_model.load_dataset_and_preprocess_data(keep_hashtag_text=keep_hashtag_text,
                                                            remove_stopwords=remove_stopwords)

        danish_topic_model.create_topic_model()

        danish_topic_model.train_topic_model()

    most_freq_topics = danish_topic_model.get_most_frequent_topics()

    print("\nThese are the most frequently represented topics in your topic model:\n")

    print(most_freq_topics)

    danish_topic_model.write_topic_visualization()

    danish_topic_model.get_closest_texts(topic_number=topic_number)

    danish_topic_model.create_most_frequent_wordclouds()

    if save_model:

        danish_topic_model.save_model(path=save_path)

    print("\nDONE! Have a nice day. :-)")


class DanishTopicFormers():

    def __init__(self,
                 dataset="danish_political_comments",
                 embedding_model="distiluse-base-multilingual-cased-v2",
                 out_dir=None
                 ):
        """Creates an instance of the DanishTopicFormers model

        Args:
            dataset (str, optional): A dataset from the Hugging Face hub. Defaults to "danish_political_comments".
            embedding_model (str, optional): A Danish model from sentence-transformers. Defaults to "distiluse-base-multilingual-cased-v2".
            out_dir (PosixPath, optional): Path to the output directory. Defaults to None.

        Raises:
            ValueError: If using a model not available from sentence-transformers.
        """

        if dataset is None:

            dataset = "danish_political_comments"

        if embedding_model is None:

            embedding_model = "quora-distilbert-multilingual"

        # Asserting that the embedding model is in the ones usable by BERTopic
        self.possible_models = ["distiluse-base-multilingual-cased-v2",
                                "stsb-xlm-r-multilingual",
                                "paraphrase-xlm-r-multilingual-v1",
                                "quora-distilbert-multilingual"]

        if embedding_model in self.possible_models:

            self.embedding_model = embedding_model

        else:

            raise ValueError(f"{embedding_model} is not a valid model.")

        self.out_dir = out_dir

        if self.out_dir is None:

            self.out_dir = setting_default_out_dir(assignment=6)  # Setting default output directory.

            print(f"\nOutput directory is not specified.\nSetting it to '{self.out_dir}'.")

        self.dataset = dataset

    def load_dataset_and_preprocess_data(self, dataset=None, keep_hashtag_text=True, remove_stopwords=False):
        """Load and preprocesses a Hugging Face dataset.

        Args:
            dataset (str, optional): A Hugging Face dataset. Defaults to None.

        Returns:
            list: List of cleaned sentences.
        """

        if dataset is None:

            warnings.warn(f"No dataset provided. Using default dataset: {self.dataset}")

            dataset = self.dataset

        data = self._load_dataset(dataset)

        self.texts = [self._cleaner(sentence["sentence"],
                                    keep_hashtag_text=keep_hashtag_text,
                                    remove_stopwords=remove_stopwords) for sentence in data["train"]]

        return self.texts

    def create_topic_model(self, embedding_model=None):
        """Creates a topic model

        Args:
            embedding_model (str, optional): embedding_model (str, optional): A multilingual model from sentence-transformers. Defaults to "distiluse-base-multilingual-cased-v2".

        """

        if embedding_model is None:

            warnings.warn(f"No embedding model provided. Using default embedding model: {self.embedding_model}")

            embedding_model = self.embedding_model

        print("\nCreating topic model")

        self.topic_model = BERTopic(embedding_model=embedding_model, calculate_probabilities=True, verbose=True)  # Creates instance of BERTopic model

        return self.topic_model

    def train_topic_model(self, texts=None):
        """Trains a topic model

        Args:
            texts (list, optional): List of texts to do topic modelling on. Defaults to None.
        """

        if texts is None:

            texts = self.texts

        print("\nTraining topic model")

        self.topics, self.probs = self.topic_model.fit_transform(texts)  # Fits the model on the text

        return

    def get_closest_texts(self, topic_number=None, number_of_texts=10, verbose=True):
        """Gets the closest texts from the corpus used to fit the model.
        

        Args:
            topic_number (int, optional): Which topic number to get closest text from. Defaults to None.
            number_of_texts (int, optional): Number of texts to return. Defaults to 10.
            verbose (bool, optional): Whether to print to terminal. Defaults to True.

        """

        if topic_number is None:

            topic_number = self.get_most_frequent_topics()["Topic"].to_list()[0]  # Taking the most frequent topic

            print(f"\n'topic_number' is not specified. Taking the most frequent topic with topic number: {topic_number}")

        topic_number_idx = np.where(np.asarray(self.topics) == topic_number)[0]  # Get all the topic indices where equal to the specified topic number

        topic_number_texts = np.asarray(self.texts)[topic_number_idx]  # Get the topic texts

        topic_number_probs = np.asarray([np.max(topic) for topic in self.probs[topic_number_idx]])  # Get the indices for the highest probabilities for the topic number of each document

        topic_number_probs_idx_sorted = np.flip(np.argsort(topic_number_probs))  # Sort the indices from highest prob to lowest prob

        topic_texts = topic_number_texts[topic_number_probs_idx_sorted][0:number_of_texts]  # Get the defined number texts closest to the topic

        topic_probs = topic_number_probs[topic_number_probs_idx_sorted][0:number_of_texts]  # Get the defined number probabilities closest to the topic

        topic_idx = topic_number_idx[topic_number_probs_idx_sorted][0:number_of_texts]  # Get the defined number of indices closest to the topic

        topic = self.topic_model.get_topic(topic_number)

        topic_words = self._get_topic_words(topic)  # Get topic words

        if verbose:

            print(f"\nExtracting the closest texts for topic number: {topic_number}")

            print(f"\nTopic Words: {' '.join(str(word) for word in topic_words)}")

            for txt, prob, idx in zip(topic_texts, topic_probs, topic_idx):

                print(f"\nText Number: {idx}, Probability: {prob}\n Text: {txt}")

        return topic_words, topic_texts, topic_probs, topic_idx

    def write_topic_visualization(self, out_dir=None):
        """Writes an html file to view a dimensionality reduced representation of the created topics.

        Args:
            out_dir (PosixPath, optional): Path to the output directory. Defaults to None.
        """

        if out_dir is None:

            out_dir = self.out_dir

        out_path = out_dir / "interactive_topics.html"

        fig = self.topic_model.visualize_topics()  # Creating interactive object

        fig.write_html(str(out_path))  # Writing object

    def create_wordcloud(self, topic_number=10, write_cloud=True, out_dir=None):
        """Creates a wordcloud from a given topic.

        Args:
            topic_number (int, optional): Topic to create wordcloud from. Defaults to 10.
            write_cloud (bool, optional): Whether to write the wordcloud. Defaults to True.
            out_dir (PosixPath, optional): Path to the output directory. Defaults to None.
        """

        topic = self.topic_model.get_topic(topic_number)

        word_freq_dict = self._get_word_freq_dict(topic)

        plt.figure(figsize=(10, 5),
                   dpi=200)

        plt.axis("off")

        plt.imshow(WordCloud(width=1000,
                             height=500,
                             background_color="black").generate_from_frequencies(word_freq_dict))

        plt.title("Topic " + str(topic_number), loc='left', fontsize=20, pad=5)

        if write_cloud:

            if out_dir is None:

                out_dir = self.out_dir / "wordclouds"

                out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f"wordcloud_topic_{str(topic_number)}"
            
            print(f"Writing wordcloud to: {str(out_path)}")

            plt.savefig(str(out_path))

    def create_most_frequent_wordclouds(self, number_of_topics=10, write_cloud=True, out_dir=None):
        """Creates wordclouds for the most frequent topics.

        Args:
            number_of_topics (int, optional): Number of topics to create wordclouds for. Defaults to 10.
            write_cloud (bool, optional): Whether to write the wordcloud. Defaults to True.
            out_dir (PosixPath, optional): Path to the output directory. Defaults to None.
        """

        if out_dir is None:

            out_dir = self.out_dir

        most_frequent_topics = self.get_most_frequent_topics(number_of_topics=number_of_topics)

        topic_numbers = most_frequent_topics["Topic"].to_list()

        for topic in topic_numbers:

            self.create_wordcloud(topic_number=topic, write_cloud=write_cloud)

    def get_most_frequent_topics(self, number_of_topics=10):
        """Gets most frequent topics.

        Args:
            number_of_topics (int, optional): Number of most frequent topics. Defaults to 10.

        Returns:
            pandas.Dataframe: Dataframe containing the topic number, the amount of texts for the topic and the name of the topic.
        """

        most_frequent_topics = self.topic_model.get_topic_info()[1:number_of_topics]

        return most_frequent_topics

    def save_model(self, path=None):
        """Saves a model.

        Args:
            path (PosixPath, optional): Path to the model directory. Defaults to None.
        """

        if path is None:

            path = self.out_dir / f"topic_model_dataset_{self.dataset}_embmodel_{self.embedding_model}"
            
            warnings.warn(f"No save path provided. Using default path: {path}")

        self.topic_model.save(path=path)

    def load_model(self, path=None):
        """Loads a model.

        Args:
            path (PosixPath, optional): Path to the model directory. Defaults to None.
        """

        if path is None:

            path = self.out_dir / f"topic_model_dataset_{self.dataset}_embmodel_{self.embedding_model}"
            
            warnings.warn(f"No load path provided. Using default path: {path}")

        self.topic_model.load(path=path)

    def _load_dataset(self, dataset):
        """Load dataset from Hugging Face.

        Args:
            dataset (str, optional): A Hugging Face dataset. Defaults to None.

        Returns:
            datasets.DatasetDict: A Hugging Face dataset dictionary.
        """

        data = load_dataset(dataset)  # Load data using Hugging Faces datasets

        return data

    def _cleaner(self, text, keep_hashtag_text=True, remove_stopwords=False):
        """Cleans a string of text. Very suitable for tweets.

        Args:
            text (str): String of text
            keep_hashtag_text (bool, optional): Remove the text following a hashtag. Defaults to True.

        Returns:
            str: Cleaned string of text.
        """

        text = re.sub("@[A-Za-z0-9]+", "", text)  # Remove @ sign

        text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)  # Remove http links

        text = " ".join(text.split())  # Split text

        text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI["en"])  # Remove Emojis

        if not keep_hashtag_text:

            text = re.sub(r'\B#\w*[a-zA-Z]+\w*', '', text)  # Removing hashtags all together

        else:

            text = text.replace("#", "").replace("_", " ")  # Remove hashtag sign but keep the text

        if remove_stopwords:
            
            stopwords = nltk.corpus.stopwords.words('danish')

            stopwords.extend([
                "ad", "af", "aldrig", "alle", "alt", "anden",
                "andet", "andre", "at", "bare", "begge", "blev", "blive", "bliver",
                "da", "de", "dem", "den", "denne", "der", "deres", "det", "dette", "dig",
                "din", "dine", "disse", "dit", "dog", "du", "efter", "ej", "eller", "en",
                "end", "ene", "eneste", "enhver", "er", "et", "far", "fem", "fik", "fire",
                "flere", "fleste", "for", "fordi", "forrige", "fra", "få", "får", "før",
                "god", "godt", "ham", "han", "hans", "har", "havde", "have", "hej", "helt", "hende",
                "hendes", "her", "hos", "hun", "hvad", "hvem", "hver", "hvilken", "hvis",
                "hvor", "hvordan", "hvorfor", "hvornår", "i", "ikke", "ind", "ingen", "intet",
                "ja", "jeg", "jer", "jeres", "jo", "kan", "kom", "komme", "kommer", "kun", "kunne",
                "lad", "lav", "lidt", "lige", "lille", "man", "mand", "mange", "med", "meget",
                "men", "mens", "mere", "mig", "min", "mine", "mit", "mod", "må", "ned", "nej",
                "ni", "nogen", "noget", "nogle", "nu", "ny", "nyt", "når", "nær", "næste", "næsten",
                "og", "også", "okay", "om", "op", "os", "otte", "over", "på", "se", "seks", "selv",
                "ser", "ses", "sig", "sige", "sin", "sine", "sit", "skal", "skulle", "som", "stor",
                "store", "syv", "så", "sådan", "tag", "tage", "thi", "ti", "til", "to", "tre", "ud",
                "under", "var", "ved", "vi", "vil", "ville",
                "vor", "vores", "være", "været"])  # extending the list of stopwords
        
            text_token_list = [word for word in text.split(' ') if word not in stopwords]  # Remove stopwords

            text = ' '.join(text_token_list)

        return text

    def _get_word_freq_dict(self, topic):
        """Creates dictionary of most frequent

        Args:
            topic (list): List of topic words with corresponding probabilities for the words

        Returns:
            word_freq_dict [dict]: Dict of topic words with corresponding probabilities for the words
        """

        words = self._get_topic_words(topic)

        probs = self._get_topic_probs(topic)

        word_freq_dict = dict(zip(words, np.exp(probs) / sum(np.exp(probs))))  # Create dict with softmax probabilities

        return word_freq_dict

    def _get_topic_words(self, topic):
        """Gets words from topic list

        Args:
            topic (list): List of topic words with corresponding probabilities for the words

        Returns:
            words [list]: list of topic words
        """

        words = [word for word, _ in topic]  # Extract words

        return words

    def _get_topic_probs(self, topic):
        """Gets probabilities from topic list

        Args:
            topic (list): List of topic words with corresponding probabilities for the words

        Returns:
            probs [list]: list of topic probabilities
        """

        probs = [prob for _, prob in topic]  # Extract probabilities

        return probs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--ds',
                        metavar="Dataset",
                        type=str,
                        help='Dataset to use from HuggingFace. Defaults to "dane".',
                        required=False)

    parser.add_argument('--em',
                        metavar="Embedding Model",
                        type=str,
                        help='One of the multilingual models available from sentence-transformers.',
                        required=False)

    parser.add_argument('--tn',
                        metavar="Topic Number",
                        type=int,
                        help='Topic number for getting the closest topics',
                        required=False)

    parser.add_argument('--od',
                        metavar="Output Directory",
                        type=str,
                        help='A path to the output directory.',
                        required=False)
    
    parser.add_argument('--sm',
                        dest="sm",
                        help='Whether to save a model or not to',
                        action="store_true")

    parser.set_defaults(sm=False)
    
    parser.add_argument('--sp',
                        metavar="Model Save Path",
                        type=str,
                        help='A path to the model output directory.',
                        required=False)

    parser.add_argument('--lm',
                        dest="lm",
                        help='Whether to load a model or not to',
                        action="store_true")
    
    parser.add_argument('--lp',
                        metavar="Model Load Path",
                        type=str,
                        help='A path to the model output directory.',
                        required=False)

    parser.set_defaults(lm=False)
    
    parser.add_argument('--kht',
                        dest="keep_hashtag_text",
                        help='Whether to keep the text of a hashtag or not to',
                        action="store_true")

    parser.set_defaults(keep_hashtag_text=False)
    
    parser.add_argument('--rs',
                        dest="remove_stopwords",
                        help='Whether to remove stopwords or not to',
                        action="store_true")

    parser.set_defaults(remove_stopwords=False)

    main(parser.parse_args())
