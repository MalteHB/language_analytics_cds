#!usr/bin/env python3

# Importing packages
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer, logging
import datasets
import numpy as np

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

    train_model = args.train

    test_model = args.test

    HFNER = HuggingFaceNamedEntityClassification(dataset=dataset,
                                                 language=language,
                                                 hf_model_path=hf_model_path,
                                                 local_model_path=local_model_path,
                                                 batch_size=batch_size,
                                                 learning_rate=learning_rate,
                                                 epochs=epochs,
                                                 task=task)  # TODO: Create download data function

    if train_model:

        HFNER.setup_training()

        HFNER.train_model()

        HFNER.evaluate_model()

        HFNER.test_model()

        if save_model:

            HFNER.save_model()

    if test_model:

        HFNER.setup_training()

        HFNER.test_model()

    if sentence is None:

        HFNER.predict(model_path=local_model_path, sentence="Ross Deans Kristensen-McLachlan er en dejlig mand, arbejder for Aarhus Universitet, og bor i Aarhus.")

    else:

        HFNER.predict(model_path=local_model_path, sentence=sentence)

    print("DONE! Have a nice day. :-)")


class HuggingFaceNamedEntityClassification():
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

        self.hf_model_path = hf_model_path

        self.local_model_path = local_model_path

        if self.local_model_path is None:

            self.local_model_path = setting_default_out_dir()

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

        logging.set_verbosity_error()
        datasets.logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_path, do_lower_case=False, strip_accents=self.strip_accents)

        self.train, self.val, self.test = self.load_and_preprocess_hf_dataset(self.dataset)

        self.label_list = self.train.features[f"{self.task}_tags"].feature.names

        self.model = AutoModelForTokenClassification.from_pretrained(self.hf_model_path, num_labels=len(self.label_list))
        logging.set_verbosity_warning()



    def load_and_preprocess_hf_dataset(self, dataset):
        """loads and preprocesses a NER-dataset from the HuggingFace dataset hub.

        Returns:
            Dataset: Training, validation and test dataset.
        """
        # Loading dane
        hf_datasets = datasets.load_dataset(dataset)

        def _tokenize_and_align_labels(examples):

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
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

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

    def test_model(self):

        def _compute_metrics(pred):

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
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

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
            compute_metrics=_compute_metrics
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
        print(results)

    def evaluate_model(self):

        print(self.trainer.evaluate())

    def save_model(self, local_model_path=None):

        if local_model_path is None:

            self.trainer.save_model(self.local_model_path)

        else:

            self.trainer.save_model(local_model_path)

    def predict(self,
                model_path=None,
                sentence=None):

        if model_path is None:

            model_path = self.local_model_path

        if self.language == "danish":

            tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False, strip_accents=False)

        else:

            tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)
        
        logging.set_verbosity_error()
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        logging.set_verbosity_warning()

        # Preprocessing sentence
        tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=True)

        input_ids = torch.tensor([tokenized_sentence])

        # Getting predictions from the model
        with torch.no_grad():

            logits = model(input_ids)

        logits = F.softmax(logits[0], dim=2)

        logits_label = torch.argmax(logits, dim=2)

        logits_label = logits_label.detach().cpu().numpy().tolist()[0]

        logits_confidence = [values[label].item() for values, label in zip(logits[0], logits_label)]


        # Joining tokens, tags and confidence probabilities of the model
        tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])

        new_tokens, new_labels, new_probs = [], [], []

        for token, label_idx, probs in zip(tokens, logits_label, logits_confidence):

            if token.startswith("##"):

                new_tokens[-1] = new_tokens[-1] + token[2:]

            else:

                if token != "[CLS]":

                    if token != "[SEP]":

                        new_labels.append(self.label_list[label_idx])

                        new_tokens.append(token)

                        new_probs.append(probs)

        print(f"Input Tokens: {' '.join(new_tokens)}",
              f"\nPredicted Entities: {' '.join(new_labels)}",
              f"\nProbabilities: {new_probs}")


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

    parser.add_argument('--train',
                        dest="train",
                        help='Whether to train a model or not to',
                        action="store_true")

    parser.set_defaults(train=False)

    parser.add_argument('--test',
                        dest="test",
                        help='Test the model',
                        action="store_true")

    parser.set_defaults(test=False)


    main(parser.parse_args())
