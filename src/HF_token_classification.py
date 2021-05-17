#!usr/bin/env python3

# Importing packages
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer, logging
import datasets
import numpy as np
import pandas as pd

from utils.utils import setting_default_out_dir


def main(args):

    # Importing arguments from the arguments parser

    dataset = args.ds

    language = args.l

    hf_model_path = args.hfmp

    local_model_path = args.lmp

    save_model = args.sm

    task = args.t

    batch_size = args.bs

    learning_rate = args.lr

    epochs = args.e

    sentence = args.s

    test_model = args.test
    
    predict = args.predict

    HFTP = HuggingFaceTokenClassification(dataset=dataset,
                                          language=language,
                                          hf_model_path=hf_model_path,
                                          local_model_path=local_model_path,
                                          batch_size=batch_size,
                                          learning_rate=learning_rate,
                                          epochs=epochs,
                                          task=task)

    if predict:
        
        if sentence is None:
    
            sentence = "Ross er en dejlig mand, som kommer fra Skotland, og underviser på Aarhus Universitet."

            print(f"\n\nINITIALISING PREDICTION OF STANDARD SENTENCE: '{sentence}'")

            HFTP.predict(model_path=local_model_path, sentence=sentence)

        else:
            print(f"\n\nINITIALISING PREDICTION OF INPUT SENTENCE: '{sentence}'")

            HFTP.predict(model_path=local_model_path, sentence=sentence)

        print("\nDONE! Have a nice day. :-)")
        
    elif test_model:
            
        print("\n\nINITIALISING EVALUATION ON TEST DATASET!")

        HFTP.setup_training()

        HFTP.test_model()
        
    else:

        print("\n\nINITIALISING TRAINING!")

        HFTP.setup_training()

        HFTP.train_model()

        print("\n\nINITIALISING EVALUATION ON VALIDATION DATASET!")

        HFTP.evaluate_model()

        if save_model:

            HFTP.save_model()

        print("\n\nINITIALISING EVALUATION ON TEST DATASET!")

        HFTP.setup_training()

        HFTP.test_model()

        if sentence is None:

            sentence = "Ross er en dejlig mand, som kommer fra Skotland, og underviser på Aarhus Universitet."

            print(f"\n\nINITIALISING PREDICTION OF STANDARD SENTENCE: '{sentence}'")

            HFTP.predict(model_path=local_model_path, sentence=sentence)

        else:
            print(f"\n\nINITIALISING PREDICTION OF INPUT SENTENCE: '{sentence}'")

            HFTP.predict(model_path=local_model_path, sentence=sentence)

            print("\nDONE! Have a nice day. :-)")


class HuggingFaceTokenClassification():
    """Takes any Danish or English token classification dataset
       from the HuggingFace's datasets-package and trains
       a small ELECTRA-model capable of predicting tokens.
    """

    def __init__(self,
                 dataset="dane",
                 language="danish",
                 hf_model_path=None,
                 local_model_path=None,
                 batch_size=8,
                 learning_rate=3e-5,
                 epochs=8,
                 task="ner"
                 ):

        # Assigning variables to the HuggingFaceTokenClassification class.
        self.hf_model_path = hf_model_path

        self.local_model_path = local_model_path

        if self.local_model_path is None:

            self.local_model_path = setting_default_out_dir(assignment=5)

        self.language = language

        if self.language is None:

            self.language = "danish"

        if self.language == "danish":

            self.hf_model_path = "Maltehb/-l-ctra-danish-electra-small-cased"

        elif self.language == "english":

            self.hf_model_path = "google/electra-small-discriminator"

        else:

            raise Exception(f"Sorry, {language} is not supported yet. \nChoose between 'danish' or 'english'")

        self.dataset = dataset

        if self.dataset is None:

            self.dataset = "dane"

        self.batch_size = batch_size

        if self.batch_size is None:

            self.batch_size = 8

        self.learning_rate = learning_rate

        if self.learning_rate is None:

            self.learning_rate = 3e-5

        self.epochs = epochs

        if self.epochs is None:

            self.epochs = 8

        self.task = task

        if self.task is None:

            self.task = "ner"

        if self.hf_model_path == "Maltehb/-l-ctra-danish-electra-small-cased":

            self.strip_accents = False

        else:

            self.strip_accents = True

        logging.set_verbosity_error()  # Decreasing verbosity
        datasets.logging.set_verbosity_error()  # Decreasing verbosity
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_path, do_lower_case=False, strip_accents=self.strip_accents)

        self.train, self.val, self.test = self.load_and_preprocess_hf_dataset(self.dataset)  # Loading and preprocessing data

        self.label_list = self.train.features[f"{self.task}_tags"].feature.names  # Creating a list of labels from the training dataset

        self.model = AutoModelForTokenClassification.from_pretrained(self.hf_model_path, num_labels=len(self.label_list))
        logging.set_verbosity_warning()



    def load_and_preprocess_hf_dataset(self, dataset):
        """loads and preprocesses a NER-dataset from the HuggingFace dataset hub.

        Returns:
            Dataset: Training, validation and test dataset.
        """
        # Loading dane
        hf_datasets = datasets.load_dataset(dataset)  # Loading dataset

        def _tokenize_and_align_labels(examples):
            """Tokenizes and aligns the labels. Adopted from:https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb

            Args:
                examples (dataset): Dataset from the datasets library.

            Returns:
                dict: A dictionary of data.
            """

            label_all_tokens = True

            tokenized_inputs = self.tokenizer(examples["tokens"], is_split_into_words=True)

            labels = []
            for i, label in enumerate(examples[f"{self.task}_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        label_ids.append(label[word_idx] if label_all_tokens else -100)
                    previous_word_idx = word_idx

                labels.append(label_ids)

            tokenized_inputs["labels"] = labels

            return tokenized_inputs

        # Applying a tokenization function on the entire dataset
        tokenized_datasets = hf_datasets.map(_tokenize_and_align_labels, batched=True)

        train, val, test = tokenized_datasets["train"], tokenized_datasets["validation"], tokenized_datasets["test"]

        return train, val, test

    def setup_training(self):
        """Function for setting up different training and data arguments.

        Returns:
            Trainer: A Trainer class from the transformers library.
        """

        self.args = TrainingArguments(
            f"test-{self.task}",
            evaluation_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
        )

        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)

        self.metric = datasets.load_metric("seqeval")

        def _compute_metrics(pred):
            """Computes metrics from predictions. Adopted from: https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb

            Args:
                pred (Dict): Predictions from the Trainer.evaluate()

            Returns:
                results_dict [Dict]: A dictionary containing the different metrics.
            """

            predictions, labels = pred
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [self.label_list[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [self.label_list[lab] for (pred, lab) in zip(prediction, label) if lab != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = self.metric.compute(predictions=true_predictions, references=true_labels)

            results_dict = {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
            return results_dict

        self.trainer = Trainer(
            self.model,
            self.args,
            train_dataset=self.train,
            eval_dataset=self.val,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=_compute_metrics
        )

        return self.trainer

    def train_model(self):

        self.trainer.train()

    def test_model(self, save_metrics=True):
        """Tests the trained model on the test dataset.
        """

        def _test_compute_metrics(pred):
            """Computes metrics from predictions. Adopted from: https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb

            Args:
                pred (Dict): Predictions from the Trainer.evaluate()

            Returns:
                results_dict [Dict]: A dictionary containing the different metrics.
            """

            predictions, labels = pred
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [self.label_list[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [self.label_list[lab] for (pred, lab) in zip(prediction, label) if lab != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = self.metric.compute(predictions=true_predictions, references=true_labels)
            
            results_dict = {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

            return results_dict

        logging.set_verbosity_error()
        model = AutoModelForTokenClassification.from_pretrained(self.local_model_path)
        logging.set_verbosity_warning()

        trainer = Trainer(
            model,
            self.args,
            train_dataset=self.train,
            eval_dataset=self.val,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=_test_compute_metrics
        )

        predictions, labels, _ = trainer.predict(self.test)
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[lab] for (pred, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        
        results_df = pd.DataFrame.from_records(results).transpose()
        
        print(results_df)
        
        if save_metrics:
            
            out_dir = setting_default_out_dir(assignment=5)
            
            csv_path = out_dir.parent / "test_metrics.csv"
            
            results_df.to_csv(csv_path)
        
        return results

    def evaluate_model(self):

        print(self.trainer.evaluate())

    def save_model(self, local_model_path=None):
        """Saves the model

        Args:
            local_model_path (str, optional): Path to the local model directory. Defaults to None.
        """

        if local_model_path is None:

            self.trainer.save_model(self.local_model_path)

        else:

            self.trainer.save_model(local_model_path)
           
    def predict(self,
                model_path=None,
                sentence=None):

        if model_path is None:

            model_path = self.local_model_path
            
        # Getting tokenizer

        if self.language == "danish":
            
            tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False, strip_accents=False)

        else:

            tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)
            
        # Getting model with temporary decreased verbosity

        logging.set_verbosity_error()
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        logging.set_verbosity_warning()

        # Preprocessing sentence
        tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=True)
        
        # Creating a torch tensor
        input_ids = torch.tensor([tokenized_sentence])

        # Getting predictions from the model with no adjustment of weights
        with torch.no_grad():

            logits = model(input_ids)

        logits = F.softmax(logits[0], dim=2)  # Applying softmax

        logits_label = torch.argmax(logits, dim=2)  # Getting highest softmax

        logits_label = logits_label.detach().cpu().numpy().tolist()[0]  # Convert labels to list

        logits_confidence = [values[label].item() for values, label in zip(logits[0], logits_label)]  # Getting the probabilities of the model


        # Joining tokens, tags and confidence probabilities of the model
        tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])  # Converting the indeces to tokens

        new_tokens, new_labels, new_probs = [], [], []

        # Collapsing tokens from word piece tokenization to actual tokens.
        for token, label_idx, probs in zip(tokens, logits_label, logits_confidence):

            if token.startswith("##"):

                new_tokens[-1] = new_tokens[-1] + token[2:]

            else:

                if token != "[CLS]":

                    if token != "[SEP]":

                        new_labels.append(self.label_list[label_idx])

                        new_tokens.append(token)

                        new_probs.append(probs)

        print(f"\n\nInput Tokens: {' '.join(new_tokens)}",
              f"\n\nPredicted Entities: {' '.join(new_labels)}",
              f"\n\nProbabilities: {new_probs}")


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

    parser.add_argument('--hfmp',
                        metavar="HuggingFace Model Path",
                        type=str,
                        help='Model to use from the HuggingFace model hub.',
                        required=False)

    parser.add_argument('--lmp',
                        metavar="Local Model Path",
                        type=str,
                        help='Path to save the model to.',
                        required=False)

    parser.add_argument('--bs',
                        metavar="Batch Size",
                        type=int,
                        help='Batch Size to the model training setup.',
                        required=False)

    parser.add_argument('--lr',
                        metavar="Learning Rate",
                        type=int,
                        help='Learning Rate for the model training setup.',
                        required=False)

    parser.add_argument('--e',
                        metavar="Epochs",
                        type=int,
                        help='Number of epochs.',
                        required=False)

    parser.add_argument('--sm',
                        metavar="Save Model",
                        type=bool,
                        help='Whether to save the model',
                        default=True)

    parser.add_argument('--t',
                        metavar="Task",
                        type=str,
                        help='What task to take from the HuggingFace Token Classification Dataset',
                        required=False)

    parser.add_argument('--s',
                        metavar="Sentence",
                        type=str,
                        help='Sentence to be predicted by the trained model.',
                        required=False)

    parser.add_argument('--test',
                        dest="test",
                        help='Test the model',
                        action="store_true")

    parser.set_defaults(test=False)
    
    parser.add_argument('--p',
                        dest="predict",
                        help='Whether to only predict a sentence.',
                        action="store_true")

    parser.set_defaults(predict=False)


    main(parser.parse_args())
