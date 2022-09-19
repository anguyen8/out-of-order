import csv
import glob
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from lxml import etree
from collections import defaultdict

import numpy as np
import random
import re

logger = logging.getLogger(__name__)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, aspect=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.aspect = aspect

    def to_json(self):
        json_obj = {"guid": self.guid,
                    "text_a": self.text_a,
                    "text_b": self.text_b,
                    "label": self.label}

        if self.aspect is not None:
            json_obj["aspect"] = self.aspect

        return json_obj


@dataclass(frozen=True)
class MultipleChoiceExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""
    def __init__(self, data_dir, task_name=None):
        self.data_dir = data_dir
        self.task_name = task_name

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class SwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""
    def get_train_examples(self):
        """See base class."""
        logger.info("LOOKING AT {} train".format(self.data_dir))
        return self._create_examples(self._read_csv(os.path.join(self.data_dir, "train.csv")), Split.train)

    def get_dev_examples(self):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(self.data_dir))
        return self._create_examples(self._read_csv(os.path.join(self.data_dir, "val.csv")), Split.dev)

    def get_test_examples(self):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(self.data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code setting!"
        )
        return self._create_examples(self._read_csv(os.path.join(self.data_dir, "test.csv")), Split.test)

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: Split):
        """Creates examples for the training and dev sets."""
        if type == Split.train and lines[0][-1] != "label":
            raise ValueError("For training, the input file must contain a label column.")

        examples = [
            MultipleChoiceExample(
                example_id=line[2],
                question=line[5],  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[4], line[4], line[4], line[4]],
                endings=[line[7], line[8], line[9], line[10]],
                label=line[11],
            )
            for line in lines[1:]  # we skip the line with the column names
        ]

        return examples


class SemEvalAtisProcessor(DataProcessor):
    """Processor for the ABSA datasets"""
    def get_train_examples(self):
        if self.task_name == "atis_ner":
            return self._create_examples_atis_ner(data_dir=self.data_dir, type=Split.train)

        return self._create_examples_semeval_and_atis(data_dir=self.data_dir, type=Split.train, task_name=self.task_name)

    def get_dev_examples(self):
        if self.task_name == "atis_ner":
            return self._create_examples_atis_ner(data_dir=self.data_dir, type=Split.dev)

        return self._create_examples_semeval_and_atis(data_dir=self.data_dir, type=Split.dev, task_name=self.task_name)

    def get_test_examples(self):
        if self.task_name == "atis_ner":
            return self._create_examples_atis_ner(data_dir=self.data_dir, type=Split.test)

        return self._create_examples_semeval_and_atis(data_dir=self.data_dir, type=Split.test, task_name=self.task_name)

    def get_labels(self):
        if self.task_name == "semeval":
            return ['positive', 'negative', 'neutral', 'conflict']

        elif self.task_name == "atis":
            dict_intent_path = self.data_dir + "atis.dict.intent.csv"
            dict_intent_file = open(dict_intent_path, "r")
            id2intent = [line.strip() for line in dict_intent_file]
            dict_intent_file.close()
            return id2intent

        elif self.task_name == "atis_ner":
            dict_slot_path = self.data_dir + "atis.dict.slots.csv"
            dict_slot_file = open(dict_slot_path, "r")
            id2slot = [line.strip() for line in dict_slot_file]
            dict_slot_file.close()
            return id2slot

        else:
            raise Exception("Invalid task %s..." % self.task_name)

    def _create_examples_atis_ner(self, data_dir: str, type: Split):

        data_path = ""
        examples = []

        if type == Split.train:
            data_path = os.path.join(data_dir, "train_with_tags.txt")
        elif type == Split.dev:
            data_path = os.path.join(data_dir, "dev_with_tags.txt")
        elif type == Split.test:
            data_path = os.path.join(data_dir, "test_with_tags.txt")

        with open(data_path, 'r', encoding='UTF-8') as fp:
            sample_id = 0
            for line in fp:
                sentence, label, _ = line.strip().split("|||||")

                guid = "%s-%s" % (type, sample_id)
                text_a = sentence
                tags = label.split(" ")

                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=tags))
                sample_id += 1

        return examples

    def _create_examples_semeval_and_atis(self, data_dir, type, task_name):

        data_path = ""
        examples = []

        if task_name == "semeval":
            if type == Split.train:
                data_path = os.path.join(data_dir, "Restaurants_Train_v2.xml")
            elif type == Split.dev:
                data_path = os.path.join(data_dir, "Restaurants_Dev_v2.xml")
            elif type == Split.test:
                data_path = os.path.join(data_dir, "Restaurants_Test_Gold.xml")

            input_data = self.get_semeval(data_path, type)
            unrolled_items = []
            total_counter = defaultdict(int)

            for e in input_data:
                for aspect, sentiment in e['aspect_sentiment']:
                    unrolled_items.append({'sentence': e['sentence'], 'aspect': aspect, 'sentiment': sentiment})
                    total_counter[sentiment] += 1

            print("total")
            print(total_counter)
            sample_id = 0

            for item in unrolled_items:
                guid = "%s-%s" % (type, sample_id)
                text_a = item['sentence']
                tag = item['sentiment']
                aspect = item['aspect']

                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=tag, aspect=aspect))
                sample_id += 1

        elif task_name == "atis":
            if type == Split.train:
                data_path = os.path.join(data_dir, "train_with_tags.txt")
            elif type == Split.dev:
                data_path = os.path.join(data_dir, "dev_with_tags.txt")
            elif type == Split.test:
                data_path = os.path.join(data_dir, "test_with_tags.txt")

            data_file = open(data_path, "r")
            for idx, line in enumerate(data_file):
                sentence, _, intent = line.strip().split("|||||")

                guid = "%s-%s" % (type, idx)
                text_a = sentence
                tag = intent

                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=tag))

        return examples

    def read_sentence14(self, file_path):
        dataset = []

        with open(file_path, 'rb') as fopen:
            raw = fopen.read()
            root = etree.fromstring(raw)
            for sentence in root:
                example = dict()
                example["sentence"] = sentence.find('text').text.lower()

                # ThangPM's Note:
                # Fix bug caused by Spacy tokenizer for this sentence:
                # "It took 100 years for Parisi to get around to making pizza (at least I don't think they ever made it before this year)...but it was worth the wait."
                if example["sentence"].find("...") != -1:
                    example["sentence"] = example["sentence"].replace("...", " ... ").strip()
                    example["sentence"] = re.sub(r"\s{2,}", " ", example["sentence"])  # If a white space " " is repeated at least 2 times or more,
                                                                                       # replace all consecutive white spaces by a single one
                    example["sentence"] = example["sentence"].strip()

                categories = sentence.find("aspectCategories")
                example["aspect_sentiment"] = []

                for c in categories:
                    aspect = c.attrib['category'].lower()
                    if aspect == 'anecdotes/miscellaneous':
                        aspect = 'misc'
                    example["aspect_sentiment"].append((aspect, c.attrib['polarity']))

                dataset.append(example)

        return dataset

    def get_semeval(self, data_path, set_type):

        semeval14_examples = self.read_sentence14(data_path)
        semeval14_examples = list(semeval14_examples)
        print("# SemEval 14 {0}: {1}".format(set_type, len(semeval14_examples)))
        print("# {}: {}".format(set_type, len(semeval14_examples)))

        return semeval14_examples


processors = {"swag": SwagProcessor,
              "semeval_atis": SemEvalAtisProcessor}


MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"race", 4, "swag", 4, "arc", 4}