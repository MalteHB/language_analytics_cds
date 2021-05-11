#!usr/bin/env python3
# TODO: Implement prediction functionality with probabilities
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

    # local_model_path = args.lmp

    # save_model = args.sm

    # task = args.t

    # batch_size = args.bs

    # learning_rate = args.lr

    # epochs = args.e

    # sentence = args.s

    # train_model = args.train

    # test_model = args.test

    danish_topic_model = MultilingualTopicFormers(dataset=dataset, embedding_model=embedding_model)

    danish_topic_model.load_dataset_and_preprocess_data()

    danish_topic_model.create_topic_model()

    danish_topic_model.train_topic_model()

    most_freq_topics = danish_topic_model.get_most_frequent_topics()

    print("\nThese are the most frequently represented topics in your topic model:\n")

    print(most_freq_topics)

    danish_topic_model.write_topic_visualization()

    topic_words, topic_texts, topic_scores, topic_idx = danish_topic_model.get_closest_texts(topic_number=topic_number)

    print(f"\nExtracting the closest texts for topic number: {topic_number}")

    print(f"\nTopic Words: {' '.join(str(word) for word in topic_words)}")

    for txt, score, idx in zip(topic_texts, topic_scores, topic_idx):

        print(f"\nText: {idx}, Score: {score}")

        print(txt)

    danish_topic_model.create_most_frequent_wordclouds()

    print("\nDONE! Have a nice day. :-)")


class MultilingualTopicFormers():

    def __init__(self,
                 dataset="danish_political_comments",
                 embedding_model="distiluse-base-multilingual-cased-v2",
                 out_dir=None
                 ):

        if dataset is None:

            dataset = "danish_political_comments"

        if embedding_model is None:

            embedding_model = "distiluse-base-multilingual-cased-v2"

        # Asserting that the embedding model is in the ones usable by BERTopic
        self.possible_models = ["distiluse-base-multilingual-cased-v2",
                                "stsb-xlm-r-multilingual",
                                "paraphrase-xlm-r-multilingual-v1"
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

    def load_dataset_and_preprocess_data(self, dataset=None):

        if dataset is None:

            warnings.warn(f"No dataset provided. Using default dataset: {self.dataset}")

            dataset = self.dataset

        data = self._load_dataset(dataset)

        self.texts = [self._cleaner(sentence["sentence"]) for sentence in data["train"]]

        return self.texts

    def create_topic_model(self, embedding_model=None, texts=None):

        if texts is None:

            texts = self.texts

        if embedding_model is None:

            warnings.warn(f"No embedding model provided. Using default embedding model: {self.dataset}")

            embedding_model = self.embedding_model

        print("\nCreating topic model")

        self.topic_model = BERTopic(embedding_model=embedding_model, calculate_probabilities=True)  # Creates instance of BERTopic model

        return self.topic_model

    def train_topic_model(self, texts=None):

        if texts is None:

            texts = self.texts

        print("\nTraining topic model")

        self.topics, self.probs = self.topic_model.fit_transform(texts)  # Fits the model on the text

        return

    def get_closest_texts(self, topic_number, number_of_texts=10):

        topic_number_idx = np.where(np.asarray(self.topics) == topic_number)[0]  # Get all the topic indices where equal to the specified topic number

        topic_number_texts = np.asarray(self.texts)[topic_number_idx]  # Get the topic texts

        topic_number_probs = np.asarray([np.max(topic) for topic in self.probs[topic_number_idx]])  # Get the indices for the highest probabilities for the topic number of each document

        topic_number_probs_idx_sorted = np.flip(np.argsort(topic_number_probs))  # Sort the indices from highest prob to lowest prob

        topic_texts = topic_number_texts[topic_number_probs_idx_sorted][0:number_of_texts]  # Get the defined number texts closest to the topic

        topic_scores = topic_number_probs[topic_number_probs_idx_sorted][0:number_of_texts]  # Get the defined number probabilities closest to the topic

        topic_idx = topic_number_idx[topic_number_probs_idx_sorted][0:number_of_texts]  # Get the defined number of indices closest to the topic

        topic = self.topic_model.get_topic(topic_number)

        topic_words = self._get_topic_words(topic)  # Get topic words

        return topic_words, topic_texts, topic_scores, topic_idx

    def write_topic_visualization(self, out_dir=None):

        if out_dir is None:

            out_dir = self.out_dir

        out_path = out_dir / "interactive_topics.html"

        fig = self.topic_model.visualize_topics()  # Creating interactive object

        fig.write_html(str(out_path))  # Writing object

    def create_wordcloud(self, topic_number=10, write_cloud=True, out_dir=None):

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

            plt.savefig(str(out_path))

    def create_most_frequent_wordclouds(self, number_of_topics=10, write_cloud=True, out_dir=None):

        if out_dir is None:

            out_dir = self.out_dir

        most_frequent_topics = self.get_most_frequent_topics(number_of_topics=number_of_topics)

        topic_numbers = most_frequent_topics["Topic"].to_list()

        for topic in topic_numbers:

            self.create_wordcloud(topic_number=topic, write_cloud=write_cloud)


    def get_most_frequent_topics(self, number_of_topics=10):

        most_frequent_topics = self.topic_model.get_topic_info()[1:number_of_topics]

        return most_frequent_topics

    def _load_dataset(self, dataset):

        data = load_dataset(dataset)  # Load data using Hugging Faces datasets

        return data

    def _cleaner(self, text, remove_hashtag_text=True):

        stopwords = nltk.corpus.stopwords.words('danish')

        stopwords.extend(["ad", "af", "aldrig", "alle", "alt", "anden", \
                          "andet", "andre", "at", "bare", "begge", "blev", "blive", "bliver", \
                          "da", "de", "dem", "den", "denne", "der", "deres", "det", "dette", "dig", \
                          "din", "dine", "disse", "dit", "dog", "du", "efter", "ej", "eller", "en", \
                          "end", "ene", "eneste", "enhver", "er", "et", "far", "fem", "fik", "fire", \
                          "flere", "fleste", "for", "fordi", "forrige", "fra", "få", "får", "før", \
                          "god", "godt", "ham", "han", "hans", "har", "havde", "have", "hej", "helt", "hende", \
                          "hendes", "her", "hos", "hun", "hvad", "hvem", "hver", "hvilken", "hvis", \
                          "hvor", "hvordan", "hvorfor", "hvornår", "i", "ikke", "ind", "ingen", "intet", \
                          "ja", "jeg", "jer", "jeres", "jo", "kan", "kom", "komme", "kommer", "kun", "kunne", \
                          "lad", "lav", "lidt", "lige", "lille", "man", "mand", "mange", "med", "meget", \
                          "men", "mens", "mere", "mig", "min", "mine", "mit", "mod", "må", "ned", "nej", \
                          "ni", "nogen", "noget", "nogle", "nu", "ny", "nyt", "når", "nær", "næste", "næsten", \
                          "og", "også", "okay", "om", "op", "os", "otte", "over", "på", "se", "seks", "selv", \
                          "ser", "ses", "sig", "sige", "sin", "sine", "sit", "skal", "skulle", "som", "stor", \
                          "store", "syv", "så", "sådan", "tag", "tage", "thi", "ti", "til", "to", "tre", "ud", \
                          "under", "var", "ved", "vi", "vil", "ville", \
                          "vor", "vores", "være", "været"])  # extending the list of stopwords

        text = re.sub("@[A-Za-z0-9]+", "", text)  # Remove @ sign

        text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)  # Remove http links

        text = " ".join(text.split())  # Split text

        text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)  # Remove Emojis TODO: REMOVE ALL EMOJIS 🤔🤔🤔🤔

        if remove_hashtag_text:

            text = re.sub(r'\B#\w*[a-zA-Z]+\w*','', text)  # Removing hashtags all together

        else:

            text = text.replace("#", "").replace("_", " ")  # Remove hashtag sign but keep the text

        text_token_list = [word for word in text.split(' ') if word not in stopwords]  # Remove stopwords

        text = ' '.join(text_token_list)

        return text

    def _get_word_freq_dict(self, topic):

        words = self._get_topic_words(topic)

        probs = self._get_topic_probs(topic)

        word_freq_dict = dict(zip(words, np.exp(probs)/sum(np.exp(probs))))  # Create dict with softmax probabilities

        return word_freq_dict

    def _get_topic_words(self, topic):

        words = [word for word, _ in topic]  # Extract words

        return words

    def _get_topic_probs(self, topic):

        probs = [prob for _, prob in topic]  # Extract probabilities

        return probs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--ds',
                        metavar="Dataset",
                        type=str,
                        help='Dataset to use from HuggingFace. Defaults to "dane".',
                        required=False)

    parser.add_argument('--l',
                        metavar="Language",
                        type=str,
                        help='Language of the dataset. Defaults to "danish".',
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
                        required=False,
                        default=10)

    parser.add_argument('--od',
                        metavar="Output Directory",
                        type=str,
                        help='A path to the output directory.',
                        required=False)

    main(parser.parse_args())
