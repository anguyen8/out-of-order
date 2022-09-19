import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from ...tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_xlm_roberta import XLMRobertaTokenizer
from ..processors.glue import glue_convert_examples_to_features, glue_output_modes, glue_processors
from ..processors.utils import InputFeatures


import random
import numpy as np
from ..processors.utils import InputExample

from src.ris import utils


logger = logging.getLogger(__name__)


@dataclass
class GlueDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    # ThangPM
    shuffle_data: bool = field(
        default=False, metadata={"help": "Shuffle dev set when evaluation"}
    )

    shuffle_type: str = field(
        default="", metadata={"help": "Specify shuffle type in [unigram, bigram, trigram]"})
        # default="shuffled", metadata={"help": "Specify shuffle type in [unigram, bigram, trigram]"})

    get_sent_embs: bool = field(
        default=False, metadata={"help": "Ask model to obtain and store sentence embeddings"}
    )

    shuffle_dir: str = field(
        default="", metadata={"help": "Specify shuffle data directory for loading"})

    model_base: str = field(
        default="", metadata={"help": "Model base (e.g., bert-base-uncased, roberta-base)"})

    masked_lm: str = field(
        default="", metadata={"help": "Masked language models for running analyzers (e.g., bert-base-uncased, RoBERTa)"})

    analyzer: str = field(
        default="", metadata={"help": "Analyzers (e.g., RIS, OccEmpty, OccZero)"})

    checkpoint: str = field(
        default="", metadata={"help": "Finetuned model at specific checkpoint"})


    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    ris = "ris"


class GlueDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int]=None,
        mode: Union[str, Split]=Split.train,
        cache_dir: Optional[str]=None,
        examples=None,
        shuffle_data=False,
        shuffle_type="shuffled",
        do_train=False,
        seed=42,
    ):
        self.args = args
        self.processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name, #+ "-mm"
            ),
        )

        label_list = self.processor.get_labels()
        if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
            RobertaTokenizer,
            RobertaTokenizerFast,
            XLMRobertaTokenizer,
        ):
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            # ThangPM's NOTES 06-14-20
            if shuffle_data:
                cached_features_file += "_" + shuffle_type

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start)
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                # FOR ROAR ONLY
                # ThangPM: 03/03/2021
                attr_type, roar_rate = args.shuffle_dir.split("/")

                if examples is None:
                    if mode == Split.dev:
                        self.examples = self.processor.get_dev_examples(args.data_dir)
                        self.attribution_dir = args.attribution_eval_dir if hasattr(args, 'attribution_eval_dir') else None
                    elif mode == Split.test:
                        self.examples = self.processor.get_test_examples(args.data_dir)
                    else:
                        # IMPORTANT: MUST ROLL BACK AFTER EXPERIMENT
                        self.examples = self.processor.get_train_examples(args.data_dir)
                        # self.examples = self.processor.get_dev_examples(args.data_dir)
                        self.attribution_dir = args.attribution_train_dir if hasattr(args, 'attribution_train_dir') else None
                else:
                    self.examples = examples

                if limit_length is not None:
                    self.examples = self.examples[:limit_length]

                # FOR ROAR ONLY

                import pickle
                from os.path import exists
                postfix = "_use_bert"   # in ["", "_baseline", "_use_bert", "_use_bert_sst2", "_use_bert_baseline"]
                lite_pickle_fb = self.attribution_dir + attr_type + "/" + "del_examples_" + roar_rate + postfix + ".pickle"
                print("********** LOADING PICKLE FILE: " + lite_pickle_fb)
                with open(lite_pickle_fb, "rb") as file_path:
                    self.examples = pickle.load(file_path)


                # ---------------------------------------------------------------------------------------
                # IMPORTANT: FOR FINETUNING COMBINATION OF NLI TASKS ONLY + EVALUATION ON ANLI (R1, R2, R3)
                # ---------------------------------------------------------------------------------------
                '''
                import ast
                if args.task_name in ["mnli", "mnli-mm"]:
                    # file_name = "train.txt"
                    # if mode == Split.dev:
                    #     file_name = "m_dev.txt" if args.task_name == "mnli" else "mm_dev.txt"
                    #
                    # input_path = "/".join(args.data_dir.split("/")[:-1]) + "/SMFA_NLI/" + file_name
                    # input_file = open(input_path, "r")

                    synthetic_task = args.shuffle_dir.lower()
                    file_name = synthetic_task + ".txt"
                    input_path = "/".join(args.data_dir.split("/")[:-1]) + "/ANLI/dev/" + file_name
                    input_file = open(input_path, "r")

                    print("********** Loading data from path: " + input_path + " **********")

                    new_examples = []
                    for line in input_file:
                        # json_obj = json.loads(line.strip())
                        json_obj = ast.literal_eval(line.strip()) # Use this because of single quote json
                        example = InputExample(guid=json_obj['guid'],
                                               text_a=json_obj['text_a'],
                                               text_b=(json_obj['text_b'] if json_obj['text_b'] != "" else None),
                                               # For RoBERTa tokenizer
                                               label=json_obj['label'])
                        new_examples.append(example)

                    self.examples = new_examples
                    input_file.close()
                '''
                # ---------------------------------------------------------------------------------------

                # ---------------------------------------------------------------------------------------
                # ThangPM's NOTES 06-14-20
                # Convert original examples to synthetic ones
                # ---------------------------------------------------------------------------------------
                # DO NOT GO THROUGH HERE IF FINETUNING
                # --> USE ORIGINAL VAL SET (NOT REMOVE ANYTHING)
                '''
                if not do_train and examples is None:
                    import json
                    examples = []

                    # bridge_dir = ""
                    if args.task_name in ["mnli", "mnli-mm"]:
                        # bridge_dir = "roberta_synthetic/5th/"
                        bridge_dir = ""
                        # bridge_dir = "10_runs_robert_large_fb_miniQ1/"
                        # bridge_dir = "10_runs_robert_large_fb_2_noun_swap/"
                        # bridge_dir = "10_runs_robert_base_fb_2_noun_swap/"
                        task_name = args.shuffle_dir.lower() # for R1, R2 and R3
                    else:
                        bridge_dir = (args.shuffle_dir + "/") if args.shuffle_dir else ""
                        task_name = args.task_name

                    bridge_dir += ((str(seed) + "/") if seed != 42 else "")
                    # bridge_dir += "bert/" + ((str(seed) + "/") if seed != 42 else "")

                    # Force to add "seed" to data path for multiple runs of extra finetuning
                    # bridge_dir += (str(seed) + "/")

                    # Prepare dev set for multiple runs, Use this --> /val_new/miniQ1_filtered_by_len/
                    # If multiple runs, Use this --> /val_new/multiple_runs/10_runs_roberta_miniQ1/
                    # Otherwise: --> /val_new/miniQ1_only_corr_preds/
                    if not shuffle_data:
                        input_path = "/".join(args.data_dir.split("/")[:-1]) + "/val_new/miniQ1_only_corr_preds/" + bridge_dir + "dev_original/" + task_name + ".txt"
                    else:
                        input_path = "/".join(args.data_dir.split("/")[:-1]) + "/val_new/miniQ1_only_corr_preds/" + bridge_dir + "dev_" + shuffle_type + "/" + task_name + ".txt"

                    print("********** Loading data from path: " + input_path + " **********")

                    input_file = open(input_path, "r")

                    for line in input_file:
                        json_obj = json.loads(line.strip())
                        example = InputExample(guid=json_obj['guid'],
                                               text_a=json_obj['text_a'],
                                               text_b=(json_obj['text_b'] if json_obj['text_b'] != "" else None), # For RoBERTa tokenizer
                                               label=json_obj['label'])
                        examples.append(example)

                    self.examples = examples
                    input_file.close()

                    # if is_synthetic:
                        # self.handle_dataset(args, is_synthetic)
                        # pass
                '''
                # ---------------------------------------------------------------------------------------

                # ---------------------------------------------------------------------------------------
                # Load synthetic datasets for extra finetuning + evaluation
                # ---------------------------------------------------------------------------------------
                '''
                import json
                examples = []

                synthetic_task = args.shuffle_dir.lower()
                subdir = "train" if do_train else "dev"
                input_path = "/".join(args.data_dir.split("/")[:-1]) + "/CoLA_synthetic/" + subdir + "/" + synthetic_task + "_synthetic" + ".txt"
                print("********** Loading data from path: " + input_path + " **********")

                input_file = open(input_path, "r")

                for line in input_file:
                    json_obj = json.loads(line.strip())
                    example = InputExample(guid=json_obj['guid'],
                                           text_a=json_obj['text_a'],
                                           text_b=(json_obj['text_b'] if json_obj['text_b'] != "" else None), # For RoBERTa tokenizer
                                           label=json_obj['label'])
                    examples.append(example)

                self.examples = examples
                input_file.close()
                '''
                # ---------------------------------------------------------------------------------------

                self.features = glue_convert_examples_to_features(
                    self.examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()

                # if examples is None:
                #     torch.save(self.features, cached_features_file)
                    # torch.save(self.features, cached_features_file + "_DN") # COLA ONLY

                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

            # To print out the distribution of dataset regarding gold labels
            distribution_dict = {}
            for feature in self.features:
                if feature.label not in distribution_dict:
                    distribution_dict[feature.label] = 0

                distribution_dict[feature.label] += 1

            print(distribution_dict)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list

    def handle_dataset(self, args, is_synthetic):
        new_examples = []
        print("Occluded RATE: " + str(args.roar))

        if is_synthetic == "attribution_only":
            attribution_list = utils.load_attribution_viz_baselines(self.attribution_dir)

        for idx, example in enumerate(self.examples):

            # IF Attribution method (RIS, Occlusion, LIME) + ROAR --> ONLY CONSIDER THIS IF CONDITION
            if is_synthetic == "attribution_only":
                occluded_sentence_a = occlude_sentence_attribution(example.text_a, count_words_removed=args.roar, attribution=attribution_list[idx], attribution_method=args.attribution_method)
                occluded_sentence_b = example.text_b  # occlude_sentence(example.text_b)
                new_examples.append(InputExample(guid=example.guid, text_a=occluded_sentence_a, text_b=occluded_sentence_b, label=example.label))

                if idx < 5:
                    print("BEFORE: " + example.text_a)
                    print("AFTER: " + occluded_sentence_a)

                continue

            # shuffled_sentence_a = shuffle_sentence(example.text_a, shuffle_type="DN")
            # shuffled_sentence_b = example.text_b    # shuffle_sentence(example.text_b)

            shuffled_sentence_a = example.text_a
            shuffled_sentence_b = shuffle_sentence(example.text_b, shuffle_type="DN")

            if idx < 5:
                print("BEFORE: " + example.text_b)
                print("AFTER: " + shuffled_sentence_b)

            if is_synthetic == "shuffled_only":
                new_examples.append(
                    InputExample(guid=example.guid, text_a=shuffled_sentence_a, text_b=shuffled_sentence_b,
                                 label=example.label))
                continue

            occluded_sentence_a = occlude_sentence(example.text_a, count_words_removed=args.roar)
            occluded_sentence_b = example.text_b  # occlude_sentence(example.text_b)

            if is_synthetic == "occluded_only":
                new_examples.append(
                    InputExample(guid=example.guid, text_a=occluded_sentence_a, text_b=occluded_sentence_b,
                                 label=example.label))
                continue

            # COLA ONLY
            # Append all synthetic sentences to the list preparing for the next step.
            # Label: 1 means original example, 0 means synthetic example.
            # new_examples.append(example)
            # if example.label == "1" and labels['0'] < labels['1']:
            #     # new_examples.append(InputExample(guid=example.guid, text_a=example.text_a, text_b=example.text_b, label="1"))
            #     new_examples.append(InputExample(guid=example.guid + "_synthetic_1", text_a=shuffled_sentence_a, text_b=shuffled_sentence_b, label="0"))
            #     # new_examples.append(InputExample(guid=example.guid + "_synthetic_2", text_a=occluded_sentence_a, text_b=occluded_sentence_b, label="0"))
            #     labels['0'] += 1

        # if is_synthetic != "shuffled_only":
        #     # random.shuffle(new_examples)
        #     shuffled_indices = np.random.permutation(len(new_examples))
        #     new_examples = [new_examples[idx] for idx in shuffled_indices]

        # Update synthetic dataset for examples
        self.examples = new_examples.copy()

        # COUNTING FOR STATISTICS - COLA ONLY
        # labels = {"0": 0, "1": 0}
        # for example in self.examples:
        #     labels[example.label] += 1
        # print("STATISTICS AFTER: " + str(labels))


def shuffle_n_grams_in_entire_sentences(words, selected_mode):
    shuffled_words = []

    # Shuffle bigram, trigram or 4-gram in the entire sentences
    start_idx = end_idx = 0
    for i in range(int(len(words) / selected_mode)):
        start_idx = i * selected_mode
        end_idx = min(start_idx + selected_mode, len(words))
        shuffled_words.append(words[start_idx + 1: end_idx] + [words[start_idx]])

    start_idx = end_idx
    # Shuffle the remaining words and add them to the shuffled list
    if start_idx + 1 < len(words):
        shuffled_words.append(words[start_idx + 1: len(words)] + [words[start_idx]])
    # If there is one word left, add it to the shuffled list
    elif start_idx + 1 == len(words):
        shuffled_words.append([words[start_idx]])

    shuffled_words = [item for sublist in shuffled_words for item in sublist]

    return shuffled_words

def shuffle_n_grams_in_part_of_sentences(words, selected_mode):
    if len(words) < selected_mode:
        selected_mode = len(words)

    # Shuffle bigram, trigram or 4-gram in the entire sentences
    ngrams_idx = np.random.randint(0, int(len(words) / selected_mode))
    start_idx = ngrams_idx * selected_mode
    end_idx = min(start_idx + selected_mode, len(words))

    n_grams = words[start_idx: end_idx]
    shuffled_indices = np.random.permutation(len(n_grams))
    while any(shuffled_indices == np.arange(0, len(n_grams))):
        shuffled_indices = np.random.permutation(len(n_grams))
    shuffled_n_grams = [n_grams[idx] for idx in shuffled_indices]

    shuffled_words = words[0: start_idx] + shuffled_n_grams + words[end_idx: len(words)]

    return shuffled_words

# Generate a shuffled sentence
def shuffle_sentence(sentence, shuffle_type="DN"):
    if sentence is None:
        return sentence

    words = sentence.split(" ")
    shuffled_words = words.copy()

    if len(shuffled_words) == 2:
        shuffled_words[0], shuffled_words[1] = shuffled_words[1], shuffled_words[0]
    elif shuffle_type == "DN":
        random.shuffle(shuffled_words)
        # shuffled_indices = np.random.permutation(len(shuffled_words))
        # while any(shuffled_indices == np.arange(0, len(shuffled_words))):
        #     shuffled_indices = np.random.permutation(len(shuffled_words))
        # shuffled_words = [shuffled_words[idx] for idx in shuffled_indices]
    else:
        choices = []

        if shuffle_type == "D2":
            choices = [2]  # Shuffle bigram
        elif shuffle_type == "D3":
            choices = [3]  # [2, 3]           # Shuffle either bigram or trigram
        elif shuffle_type == "D4":
            choices = [4]  # [2, 3, 4]        # Shuffle either bigram or trigram or 4-gram
        elif shuffle_type == "D5":
            choices = [2, 3, 4, -1]  # Shuffle either bigram or trigram or 4-gram or remove randomly 1 word
        elif shuffle_type == "D6":
            choices = [2, 3, 4, -1, -2]  # Shuffle either bigram or trigram or 4-gram or remove randomly 1-2 words

        # selected_mode = random.choice(choices)
        selected_idx = np.random.randint(0, len(choices))
        selected_mode = choices[selected_idx]

        # Remove randomly 1-2 words
        if selected_mode < 0:
            return occlude_sentence(sentence, count_words_removed=abs(selected_mode))

        # Shuffle bigram, trigram or 4-gram in the entire sentences
        # shuffled_words = shuffle_n_grams_in_entire_sentences(words, selected_mode=selected_mode)
        shuffled_words = shuffle_n_grams_in_part_of_sentences(words, selected_mode=selected_mode)

    return " ".join(shuffled_words)

# Randomly remove a word in a sentence
def occlude_sentence(sentence, count_words_removed=None):
    if sentence is None:
        return sentence

    words = sentence.split(" ")

    if len(words) == 1:
        return words[0]

    if count_words_removed is None:
        # count_words_removed = random.randrange(1, len(words))
        count_words_removed = np.random.randint(1, len(words))
    elif count_words_removed < 1:
        count_words_removed = (int)(count_words_removed * len(words))

    for i in range(count_words_removed):
        # removed_index = random.randrange(0, len(words))
        removed_index = np.random.randint(0, len(words))
        del words[removed_index]

    return " ".join(words)

# Randomly remove a word in a sentence
def occlude_sentence_attribution(sentence, count_words_removed, attribution, attribution_method):
    if sentence is None:
        return sentence

    words = sentence.split(" ")

    if len(words) == 1:
        return words[0]

    # Compute number of words to remove
    count_words_removed = (int)(count_words_removed * len(words))

    # Get attribution scores of each word in the sentence
    if attribution_method == "Occlusion":
        attribution_scores = attribution.get_occlusion_empty_scores()
    elif attribution_method == "LIME":
        attribution_scores = attribution.get_lime_scores()
    else:
        attribution_scores = attribution.get_ris_scores()

    # Number of words in the sentence must be equal to size of attribution scores
    assert len(words) == len(attribution_scores)

    # Remove words with highest attribution scores
    for i in range(count_words_removed):
        removed_index = attribution_scores.index(max(attribution_scores))
        del words[removed_index]
        del attribution_scores[removed_index]

    return " ".join(words)
