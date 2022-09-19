import nlp
from data_processor import *
import spacy
import nltk
import json

from os import makedirs, mkdir, listdir
from os.path import isfile, join, exists

from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from texthelper.shuffler import *

from collections import Counter
import ast
import glob
from tqdm import tqdm


def test_nlp():
    # Print all the available datasets
    print([dataset.id for dataset in nlp.list_datasets()])

    # Load a dataset and print the first examples in the training set
    squad_dataset = nlp.load_dataset('squad')
    print(squad_dataset['train'][0])

    # List all the available metrics
    print([metric.id for metric in nlp.list_metrics()])

    # Load a metric
    squad_metric = nlp.load_metric('squad')


def test_swag(mode: Split = Split.train,):
    data_dir = "datasets/swag/data/"
    processor = processors["swag"](data_dir=data_dir)

    nlp = spacy.load("en_core_web_sm")

    # Get dev examples
    if mode == Split.train:
        examples = processor.get_train_examples()
    elif mode == Split.dev:
        examples = processor.get_dev_examples()
    else:
        examples = processor.get_test_examples()

    sent_dict = {}
    skipped_indices = [2198, 2322, 2478, 2479, 2736, 7842, 13575, 18670]
    shuffled_examples = []

    for idx, example in enumerate(examples):
        assert len(set(example.contexts)) == 1

        # Sentence splitting by SpaCy but using nltk instead
        # doc = nlp(example.contexts[0])
        # sentences = [sent.text for sent in doc.sents]

        # FOR CHECKING CONTEXT
        sentences = nltk.sent_tokenize(example.contexts[0])

        # SKIP examples incorrectly splitted by nltk sentence splitter.
        if idx in skipped_indices:
            sentences = [example.contexts[0]]

        if len(sentences) not in sent_dict:
            sent_dict[len(sentences)] = 0
        sent_dict[len(sentences)] += 1

        if len(sentences) > 1:
            print("Original sentence " + str(idx) + ": " + str(example.contexts[0]))
            print("Splitted sentences: ")
            [print("\t- " + sent) for sent in sentences]
            print("")
        else:
            shuffled_examples.append(example)

        # FOR CHECKING ENDINGS
        # for ending in example.endings:
        #     sentences = nltk.sent_tokenize(ending)
        #
        #     if len(sentences) not in sent_dict:
        #         sent_dict[len(sentences)] = 0
        #     sent_dict[len(sentences)] += 1
        #
        #     if len(sentences) > 1:
        #         print("Original sentence " + str(idx) + ": " + str(example))
        #         print("Splitted sentences: ")
        #         [print(sent) for sent in sentences]
        #         print("\n")

    print(sent_dict)


def test_semeval_and_atis(mode: Split = Split.train, generate_output: bool = True):
    tasks = {"semeval": "datasets/semeval2014/",
             "atis": "datasets/atis/",
             "atis_ner": "datasets/atis/"}

    all_sent_dicts = {}

    for task, data_dir in tasks.items():

        print("********** PROCESSING TASK " + task + " **********")

        processor = processors["semeval_atis"](data_dir=data_dir, task_name=task)

        # Get dev examples
        if mode == Split.train:
            examples = processor.get_train_examples()
        elif mode == Split.dev:
            examples = processor.get_dev_examples()
        else:
            examples = processor.get_test_examples()

        sent_dict = {}

        for idx, example in enumerate(examples):
            sentences = nltk.sent_tokenize(example.text_a)

            if len(sentences) == 2 and len(sentences[-1]) == 1:
                sentences = [example.text_a]

            if len(sentences) not in sent_dict:
                sent_dict[len(sentences)] = 0
            sent_dict[len(sentences)] += 1

            if len(sentences) > 1:
                print("Original sentence " + str(idx) + ": " + str(example.text_a))
                print("Splitted sentences: ")
                [print("\t- " + sent) for sent in sentences]
                print("")

        all_sent_dicts[task] = sent_dict
        # print(sent_dict)

    # Print out all results
    [print(key + " : " + str(value)) for key, value in all_sent_dicts.items()]


def print_example_glue(n_examples=10):
    # Please pick one among the available configs:
    # ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']

    glue = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli']
    # glue = ['stsb']

    for dataset in glue:
        print("********** " + dataset + " **********")
        dataset = nlp.load_dataset('glue', dataset)
        examples = dataset['validation'][:n_examples] if 'validation' in dataset else dataset['val'][:n_examples]
        for key, value in examples.items():
            print(key + " : " + str(value))

        print("\n")
def print_example_superglue(n_examples=10):
    # Please pick one among the available configs:
    # ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc', 'wsc.fixed', 'axb', 'axg']

    superglue = ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc', 'wsc.fixed']
    for dataset in superglue:
        print("********** " + dataset + " **********")
        dataset = nlp.load_dataset('super_glue', dataset)
        examples = dataset['validation'][:n_examples]
        for key, value in examples.items():
            print(key + " : " + str(value))

        print("\n")
def print_remaining_datasets(n_examples=5):
    others = ['snli', 'squad', 'squad_v2']

    for dataset in others:
        print("********** " + dataset + " **********")
        dataset = nlp.load_dataset(dataset)
        examples = dataset['validation'][:n_examples]
        for key, value in examples.items():
            print(key + " : " + str(value))

        print("\n")


shuffle_rules = {"cola": "sentence",
                 "sst2": "sentence",
                 "mrpc": "sentence1",
                 "qqp": "question1",
                 "stsb": "sentence1",
                 "mnli_mismatched": "hypothesis",
                 "mnli_matched": "hypothesis",
                 "qnli": "question",
                 "rte_g": "sentence2", # hypothesis
                 "wnli": "sentence2",

                 "anli": "text_b",
                 "r1": "hypothesis",
                 "r2": "hypothesis",
                 "r3": "hypothesis",

                 "boolq": "question",
                 "cb": "hypothesis",
                 "copa": "premise", # context
                 "multirc": "question",
                 "record": "query", # question
                 "rte_sg": "hypothesis",
                 "wic": "sentence1",
                 "wsc": "text", # Shuffle only one sentence including span 2

                 "snli": "hypothesis",
                 "squad": "question",
                 "squad_v2": "question"}


def print_out_task_keys():
    glue = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli']
    superglue = ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc']
    stanford = ['snli', 'squad', 'squad_v2']

    task_keys = {}

    for idx, dataset_name in enumerate(glue + superglue + stanford):

        group = "glue"
        if idx >= len(glue):
            group = "super_glue"
        if idx >= len(glue) + len(superglue):
            group = "stanford"

        print("********** " + dataset_name + " **********")
        dataset = nlp.load_dataset(dataset_name) if group == "stanford" else nlp.load_dataset(group, dataset_name)
        examples = dataset['validation'][:] if 'validation' in dataset else dataset['val'][:]
        if dataset_name == "rte":
            dataset_name = "rte_g" if group == "glue" else "rte_sg"

        task_keys[dataset_name] = [key for key in examples.keys()]

    [print(key + " : " + str(value)) for key, value in task_keys.items()]
def get_task_keys(task_name):
    task_keys = {"cola" : {'text_a': 'sentence', 'label': 'label', 'guid': 'idx'},
                 "sst2" : {'text_a': 'sentence', 'label': 'label', 'guid': 'idx'},
                 "mrpc" : {'text_a': 'sentence1', 'text_b': 'sentence2', 'label': 'label', 'guid': 'idx'},
                 "qqp" : {'text_a': 'question1', 'text_b': 'question2', 'label': 'label', 'guid': 'idx'},
                 "stsb" : {'text_a': 'sentence1', 'text_b': 'sentence2', 'label': 'label', 'guid': 'idx'},
                 "mnli_mismatched" : {'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label', 'guid': 'idx'},
                 "mnli_matched" : {'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label', 'guid': 'idx'},
                 "qnli" : {'text_a': 'question', 'text_b': 'sentence', 'label': 'label', 'guid': 'idx'},
                 "rte_g" : {'text_a': 'sentence1', 'text_b': 'sentence2', 'label': 'label', 'guid': 'idx'},
                 "wnli" : {'text_a': 'sentence1', 'text_b': 'sentence2', 'label': 'label', 'guid': 'idx'},

                 "anli": {'text_a': 'text_a', 'text_b': 'text_b', 'label': 'label', 'guid': 'guid'},
                 "r1": {'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label', 'guid': 'uid'},
                 "r2": {'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label', 'guid': 'uid'},
                 "r3": {'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label', 'guid': 'uid'},

                 # The below tasks (except for SNLI) are handled directly in their corresponding projects:
                 # jiant for SuperGLUE and transformer for SQuADv1-2
                 # SNLI will be handled with the above tasks.

                 "boolq" : ['question', 'passage', 'idx', 'label'],
                 "cb" : ['premise', 'hypothesis', 'idx', 'label'],
                 "copa" : ['premise', 'choice1', 'choice2', 'question', 'idx', 'label'],
                 "multirc" : ['paragraph', 'question', 'answer', 'idx', 'label'],
                 "record" : ['passage', 'query', 'entities', 'answers', 'idx'],
                 "rte_sg": ['premise', 'hypothesis', 'idx', 'label'],
                 "wic" : ['word', 'sentence1', 'sentence2', 'start1', 'start2', 'end1', 'end2', 'idx', 'label'],
                 "wsc" : ['text', 'span1_index', 'span2_index', 'span1_text', 'span2_text', 'idx', 'label'],

                 "snli" : {'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label'},
                 "squad" : ['id', 'title', 'context', 'question', 'answers'],
                 "squad_v2" : ['id', 'title', 'context', 'question', 'answers']
                 }

    return task_keys[task_name]


def generate_input_example(example, shuffle_key, shuffle_value, task_name, group_name, labels, seed=42):
    glue = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli']
    tasks_support = glue + ['snli'] + ['anli', 'r1', 'r2', 'r3']

    # For synthetic SQuADv2 ONLY
    if task_name == "squad_v2":
        return generate_input_example_synthetic_squad_v2(example, shuffle_value)

    if task_name not in tasks_support or group_name == "super_glue":
        return None, None

    task_keys = get_task_keys((task_name + "_g") if task_name == "rte" else task_name)

    text_a = example[task_keys["text_a"]]
    text_b = example[task_keys["text_b"]] if "text_b" in task_keys else ""
    label = labels[int(example[task_keys["label"]])] if len(labels) > 0 else str(example[task_keys["label"]])
    guid = example[task_keys["guid"]] if "guid" in task_keys else ""

    original_input_example = InputExample(text_a=text_a, text_b=text_b, label=label, guid=guid)

    if shuffle_key == task_keys["text_a"]:
        # text_b = shuffle_text(text_b, type=ShuffleType.UNIGRAM, keep_punctuation=True, seed=seed)               # ONLY USED FOR SHUFFLING 2 SENTENCES
        shuffled_input_example = InputExample(text_a=shuffle_value, text_b=text_b, label=label, guid=guid)
    else:
        # text_a = shuffle_text(text_a, type=ShuffleType.UNIGRAM, keep_punctuation=True, seed=seed)               # ONLY USED FOR SHUFFLING 2 SENTENCES
        shuffled_input_example = InputExample(text_a=text_a, text_b=shuffle_value, label=label, guid=guid)

    return original_input_example, shuffled_input_example


def generate_input_example_synthetic_squad_v2(example, shuffle_value):

    original_input_example = copy.deepcopy(example)
    shuffled_input_example = copy.deepcopy(example)
    shuffled_input_example["question"] = shuffle_value

    return original_input_example, shuffled_input_example


def process_sentence_input(task_name, input_sentence, idx):

    # ThangPM's NOTES 07-28-20
    # Skip means to prevent examples incorrectly splitted by tokenizer from being removed.
    # For those tasks with empty list: SKIP ALL
    # For other tasks not in this skipped list --> NO multi-sentence inputs (GOOD)
    skipped_indices = {'mnli_mismatched': [268, 2200, 2268, 3098, 4523, 6007, 6813, 8105, 8212, 9516, 9770],
                       'mnli_matched': [477, 1740, 1873, 2728, 2780, 3174, 3758, 4772, 6195, 6597, 6979, 7039, 7883, 8981, 9246, 9438, 9736],
                       'multirc': [i for i in range(188, 195)] + [i for i in range(1273, 1282)],
                       'record': [6299],
                       'squad': [2282, 10082],
                       'squad_v2': [1474, 1779, 1816, 3318, 4367, 4679],
                       'sst2': [],
                       'mrpc': [],
                       'stsb': [],
                       'qnli': [],
                       'rte': [], # includes rte_g and rte_sg
                       'wic': [],
                       'snli': [],
                       }

    skip_removal = False
    sentences = nltk.sent_tokenize(input_sentence)

    # ThangPM's NOTE
    # CORRECT TOKENIZER *AUTOMATICALLY*
    # This approach currently affects some examples in QQP. Below is an example:
    # Which is better for automation testing: Selenium or CodedUI? Why?
    if len(sentences) == 2:
        if len(sentences[0]) == 1 or len(sentences[0].split(" ")) == 1 or \
                len(sentences[-1]) == 1 or len(sentences[-1].split(" ")) == 1:
            skip_removal = True
            sentences = [input_sentence]

    # CORRECT TOKENIZER *MANUALLY*
    # SKIP examples incorrectly splitted by nltk sentence splitter.
    if task_name in skipped_indices:
        if idx in skipped_indices[task_name] or len(skipped_indices[task_name]) == 0:
            sentences = [input_sentence]

    if len(sentences) > 1:
        # print("Original sentence " + str(idx) + ": " + str(input_sentence))
        # print("Splitted sentences: ")
        # [print("\t- " + sent) for sent in sentences]
        # print("")
        return True, sentences, skip_removal

    return False, sentences, skip_removal


def general_all_shuffled_datasets_from_scratch(generate_output=False, base_dir="preprocessed/", seed=42):
    glue = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli']
    superglue = ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc']
    stanford = ['snli', 'squad', 'squad_v2']
    anli = ['anli', "r1", "r2", "r3"]

    all_sent_dicts, get_back_dicts = {}, {}
    dev_sets = {}
    short_sent_dict = {}
    swapped_stats = {}

    # Used for ONLY 2 tasks: MultiRC and ReCoRD
    removed_indices = {'multirc': [], 'record': []}

    for idx, task_name in enumerate(glue + superglue + stanford + anli):

        # if task_name != "mrpc" and task_name != "rte":
        #     continue

        # For MiniQ1 ONLY
        # if task_name not in ["cola", "rte", "mrpc", "sst2", "qqp", "qnli", "stsb"]:
        # if task_name not in ["rte", "qqp"]:
        # if task_name not in ["qqp", "qnli", "stsb"]:
        # if task_name not in ["rte", "mrpc", "sst2", "qqp", "qnli"]:
        if task_name not in ["mrpc", "qqp"]:
            continue

        # if task_name not in ["stsb"]:
        #     continue
        # if task_name not in ["squad_v2"]:
        #     continue

        group = "glue"
        if idx >= len(glue):
            group = "super_glue"
        if idx >= len(glue) + len(superglue):
            group = "stanford"
        if idx >= len(glue) + len(superglue) + len(stanford):
            group = "anli"
        labels = []

        if task_name == "rte":
            shuffle_key = shuffle_rules[task_name + "_g"] if group == "glue" else shuffle_rules[task_name + "_sg"]
        else:
            shuffle_key = shuffle_rules[task_name]

        sent_dict, get_back = {}, []
        dev_set = {"original": [], "shuffled": [], "shuffled_bigram": [], "shuffled_trigram": [], "swapped_nouns": []}
        swapped_stats[task_name] = {"swap_2_propNouns": {}, "swap_2_nouns": {}, "swap_2_nounPhrases": {}, "unchanged": {}}

        if task_name in anli:
            new_examples = []
            input_dir = "datasets/smfa_nli/anli/"
            file_name = "dev" #"test"

            if task_name == "anli":
                file_path = input_dir + "full_unfiltered/" + file_name + ".txt"
            elif task_name == "r1":
                file_path = input_dir + "r1/" + file_name + ".jsonl"
            elif task_name == "r2":
                file_path = input_dir + "r2/" + file_name + ".jsonl"
            elif task_name == "r3":
                file_path = input_dir + "r3/" + file_name + ".jsonl"

            input_file = open(file_path, "r")
            for line in input_file:
                example = ast.literal_eval(line.strip())
                new_examples.append(example)
            input_file.close()
        else:
            # if task_name not in anli:
            #     continue

            print("********** " + task_name + " **********")
            dataset = nlp.load_dataset(task_name) if group == "stanford" else nlp.load_dataset(group, task_name)
            # dataset = dataset['train']
            dataset = dataset['validation'] if 'validation' in dataset else dataset['val']
            examples = dataset[:]
            assert shuffle_key in examples.keys()

            # ThangPM's NOTE 07-30-20
            # Only these tasks need to convert labels from integers to text
            if task_name in ["mnli_matched", "mnli_mismatched", "qnli", "rte", "snli"]:
                labels = dataset.features['label'].names

            if task_name == "squad_v2":
                print("Number of negative examples of SQuADv2: " + str(sum([len(example['text']) == 0 for example in examples['answers']])))

            # ThangPM's NOTE 07-28-20
            # Convert a dictionary (examples) with lists to a list of
            # dictionaries (new_examples) corresponding to each example.
            new_examples = []
            all_values = [examples[key] for key in examples.keys()]
            for idx in range(len(all_values[0])):
                input_dict = {}
                for idx2, key in enumerate(examples.keys()):
                    input_dict[key] = all_values[idx2][idx]
                new_examples.append(input_dict)

        for idx2, example in tqdm(enumerate(new_examples)):

            # FOR SQUADv2 ONLY
            # Retain only adversarial examples as discussed with Dr. Nguyen
            if task_name == "squad_v2" and len(example["answers"]["text"]) > 0:
                continue

            item = example[shuffle_key]
            is_multi_sent_input, sentences, skip_removal = process_sentence_input(task_name, item, idx2)

            if len(sentences) not in sent_dict:
                sent_dict[len(sentences)] = 0
            sent_dict[len(sentences)] += 1

            if skip_removal:
                get_back.append(item)

            if is_multi_sent_input:
                # ThangPM's NOTES 07-28-20
                # Store removal indices for MultiRC and ReCoRD tasks
                if task_name in removed_indices:
                    removed_indices[task_name].append(item)
            else:
                # ***************** SHUFFLE EXAMPLES HERE *****************
                # ThangPM's NOTE 07-27-20
                # SuperGLUE and SQuAD will be handled in their own projects (jiant and finetuning)
                if task_name in glue + ["snli"] + ["squad_v2"] + anli:

                    # IMPORTANT: For mini_Q1 --> Need to discuss whether follow this rule to remove short sentences
                    #            having less than 4 tokens for the remaining tasks for the ICLR paper or not.
                    # ------------------------------------------------------------------------------------------
                    tokens = sentences[0].split(" ")
                    if len(tokens) <= 3:
                        if task_name not in short_sent_dict:
                            short_sent_dict[task_name] = []
                        id_keys = ["idx", "id", "guid", "uid"]
                        short_sent_dict[task_name].append(example[key] for key in id_keys if key in example) # example['idx'] if 'idx' in example else example['id']
                        print(task_name + ": Cannot shuffle short sentence: " + sentences[0])
                        continue
                    # ------------------------------------------------------------------------------------------

                    shuffled_unigram_sentence = shuffle_text(sentences[0], type=ShuffleType.UNIGRAM, keep_punctuation=True, seed=seed)
                    shuffled_bigram_sentence = shuffle_text(sentences[0], type=ShuffleType.BIGRAM, keep_punctuation=True, seed=seed)
                    shuffled_trigram_sentence = shuffle_text(sentences[0], type=ShuffleType.TRIGRAM, keep_punctuation=True, seed=seed)

                    # Swap Noun phrases:
                    '''
                    swapped_sentence, swapped = swap_two_nouns(sentences[0], keep_punctuation=True, seed=seed, print_log=False)
                    task_keys = get_task_keys((task_name + "_g") if task_name == "rte" else task_name)
                    exp_label = new_examples[idx2][task_keys["label"]]
                    guid = new_examples[idx2][task_keys["guid"]]

                    if swapped > 0:
                        if swapped == 1:
                            if exp_label not in swapped_stats[task_name]["swap_2_propNouns"]:
                                swapped_stats[task_name]["swap_2_propNouns"][exp_label] = []
                            swapped_stats[task_name]["swap_2_propNouns"][exp_label].append(guid)
                        elif swapped == 2:
                            if exp_label not in swapped_stats[task_name]["swap_2_nouns"]:
                                swapped_stats[task_name]["swap_2_nouns"][exp_label] = []
                            swapped_stats[task_name]["swap_2_nouns"][exp_label].append(guid)
                        elif swapped == 2:
                            if exp_label not in swapped_stats[task_name]["swap_2_nouns"]:
                                swapped_stats[task_name]["swap_2_nouns"][exp_label] = []
                            swapped_stats[task_name]["swap_2_nouns"][exp_label].append(guid)
                    else:
                        if exp_label not in swapped_stats[task_name]["unchanged"]:
                            swapped_stats[task_name]["unchanged"][exp_label] = []
                        swapped_stats[task_name]["unchanged"][exp_label].append(guid)

                    original_sent, swapped_sent = generate_input_example(new_examples[idx2], shuffle_key, swapped_sentence, task_name, group, labels)
                    '''

                    original_sent, shuffled_unigram_sent = generate_input_example(new_examples[idx2], shuffle_key, shuffled_unigram_sentence, task_name, group, labels, seed)
                    original_sent, shuffled_bigram_sent = generate_input_example(new_examples[idx2], shuffle_key, shuffled_bigram_sentence, task_name, group, labels)
                    original_sent, shuffled_trigram_sent = generate_input_example(new_examples[idx2], shuffle_key, shuffled_trigram_sentence, task_name, group, labels)

                    dev_set["original"].append(original_sent)
                    dev_set["shuffled"].append(shuffled_unigram_sent)
                    dev_set["shuffled_bigram"].append(shuffled_bigram_sent)
                    dev_set["shuffled_trigram"].append(shuffled_trigram_sent)
                    # dev_set["swapped_nouns"].append(swapped_sent)

        if group != "super_glue":
            dev_sets[task_name] = dev_set

        all_sent_dicts[task_name + "_" + shuffle_key] = sent_dict
        get_back_dicts[task_name + "_" + shuffle_key] = get_back

    # Print out all results
    [print(key + " : " + str(value)) for key, value in all_sent_dicts.items()]
    # [print(key + " : " + str(value)) for key, value in get_back_dicts.items()]

    for key, values in get_back_dicts.items():
        print("--- " + key + " --- ")
        for value in values:
            print("\t- " + value)

    for task_name, values in removed_indices.items():
        print(set(values))

    # print("SWAPPED 2 NOUNS STATS")
    # print(swapped_stats)

    # Write input examples to files for GLUE + SNLI
    if generate_output:

        # IMPORTANT: For mini_Q1
        # ------------------------------------------------------------------------------------------------------------------------
        tasks_indices_dict = {}

        # Original files are:
        # 1. list_tasks_indices_roberta.txt
        # 2. list_tasks_indices_albert.txt
        # 3. list_tasks_indices_bert.txt
        if exists("list_tasks_indices_bert.txt"):

            input_file = open("list_tasks_indices_bert.txt", "r")
            for line in input_file:
                task_name, correct_preds = line.split("||")
                tasks_indices_dict[task_name] = ast.literal_eval(correct_preds)

                full_indices = []
                min_size = 0
                for key, values in tasks_indices_dict[task_name].items():
                    if task_name not in short_sent_dict:
                        min_size = len(values) # There are no short sentences for this task -> min_size is length of any of 2 lists of labels
                        break

                    tasks_indices_dict[task_name][key] = list((Counter(values) - Counter(short_sent_dict[task_name])).elements())
                    if min_size == 0 or min_size > len(tasks_indices_dict[task_name][key]):
                        min_size = len(tasks_indices_dict[task_name][key])

                for values in tasks_indices_dict[task_name].values():
                    full_indices.extend(random.sample(values, min_size))

                full_indices = sorted(full_indices)
                tasks_indices_dict[task_name] = full_indices
        # ------------------------------------------------------------------------------------------------------------------------

        for task_name, dev_set in dev_sets.items():
            original_examples, shuffled_examples, shuffled_bigram_examples, \
            shuffled_trigram_examples, swapped_nouns_examples = dev_set["original"], dev_set["shuffled"], dev_set["shuffled_bigram"], \
                                                                dev_set["shuffled_trigram"], dev_set["swapped_nouns"]

            if len(original_examples) == 0:
                continue

            # Rename these tasks to match transformer_gp2 project
            if task_name == "mnli_matched":
                file_name = "mnli"
            elif task_name == "mnli_mismatched":
                file_name = "mnli-mm"
            elif task_name == "stsb":
                file_name = "sts-b"
            elif task_name == "sst2":
                file_name = "sst-2"
            else:
                file_name = task_name

            # data_dir_original = base_dir + str(seed) + "/dev_original/"
            data_dir_shuffled = base_dir + str(seed) + "/dev_shuffled/"
            # data_dir_shuffled_bigram = base_dir + str(seed) + "/dev_shuffled_bigram/"
            # data_dir_shuffle_trigram = base_dir + str(seed) + "/dev_shuffled_trigram/"
            # data_dir_swapped_nouns = base_dir + str(seed) + "/dev_swapped_nouns/"

            # if not exists(data_dir_original): makedirs(data_dir_original)
            if not exists(data_dir_shuffled): makedirs(data_dir_shuffled)
            # if not exists(data_dir_shuffled_bigram): makedirs(data_dir_shuffled_bigram)
            # if not exists(data_dir_shuffle_trigram): makedirs(data_dir_shuffle_trigram)
            # if not exists(data_dir_swapped_nouns): makedirs(data_dir_swapped_nouns)

            # original_output = open(data_dir_original + file_name + ".txt", "w")
            shuffled_output = open(data_dir_shuffled + file_name + ".txt", "w")
            # shuffled_bigram_output = open(data_dir_shuffled_bigram + file_name + ".txt", "w")
            # shuffled_trigram_output = open(data_dir_shuffle_trigram + file_name + ".txt", "w")
            # swapped_nouns_output = open(data_dir_swapped_nouns + file_name + ".txt", "w")

            # for original_example, shuffled_example, shuffled_bigram_example, shuffled_trigram_example, swapped_nouns_example in \
            #         zip(original_examples, shuffled_examples, shuffled_bigram_examples, shuffled_trigram_examples, swapped_nouns_examples):

            for original_example, shuffled_example, shuffled_bigram_example, shuffled_trigram_example in \
                    zip(original_examples, shuffled_examples, shuffled_bigram_examples, shuffled_trigram_examples):

                if (task_name not in tasks_indices_dict) or \
                    (len(tasks_indices_dict[task_name]) > 0 and original_example.guid in tasks_indices_dict[task_name]):

                    # if isinstance(original_example, dict):
                    #     original_output.write(str(original_example) + "\n")
                    # else:
                    #     original_output.write(json.dumps(original_example.to_json()) + "\n")

                    if isinstance(shuffled_example, dict):
                        shuffled_output.write(str(shuffled_example) + "\n")
                    else:
                        shuffled_output.write(json.dumps(shuffled_example.to_json()) + "\n")

                    # if isinstance(shuffled_bigram_example, dict):
                    #     shuffled_bigram_output.write(str(shuffled_bigram_example) + "\n")
                    # else:
                    #     shuffled_bigram_output.write(json.dumps(shuffled_bigram_example.to_json()) + "\n")
                    #
                    # if isinstance(shuffled_trigram_example, dict):
                    #     shuffled_trigram_output.write(str(shuffled_trigram_example) + "\n")
                    # else:
                    #     shuffled_trigram_output.write(json.dumps(shuffled_trigram_example.to_json()) + "\n")

                    # if isinstance(swapped_nouns_example, dict):
                    #     swapped_nouns_output.write(str(swapped_nouns_example) + "\n")
                    # else:
                    #     swapped_nouns_output.write(json.dumps(swapped_nouns_example.to_json()) + "\n")

            # original_output.close()
            shuffled_output.close()
            # shuffled_bigram_output.close()
            # shuffled_trigram_output.close()
            # swapped_nouns_output.close()


def general_all_shuffled_datasets_from_files(generate_output=False, base_dir="preprocessed/", seed=42, file_name=None, two_noun_swap=False):

    glue = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli']
    superglue = ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc']
    stanford = ['snli', 'squad', 'squad_v2']
    anli = ["r1", "r2", "r3"] # 'anli',

    dev_sets = {}
    swapped_stats = {}
    tasks_indices_dict = {}

    # Filter examples which are not selected (i.e., not existed in the list_tasks_indices.txt)
    if exists(file_name):

        input_file = open(file_name, "r")
        for line in input_file:
            task_name, correct_preds = line.split("||")
            tasks_indices_dict[task_name] = ast.literal_eval(correct_preds)

            full_indices = []
            min_size = len(
                list(tasks_indices_dict[task_name].values())[0])  # min_size is length of any of 2 lists of labels

            for values in tasks_indices_dict[task_name].values():
                full_indices.extend(random.sample(values, min_size))

            full_indices = sorted(full_indices)
            tasks_indices_dict[task_name] = full_indices

    for idx, task_name in enumerate(glue + superglue + stanford + anli):

        # For MiniQ1 ONLY
        # if task_name not in ["cola", "rte", "mrpc", "sst2", "qqp", "qnli", "stsb"]:
        # if task_name not in ["stsb"]:
        if task_name not in ["rte"]:
        # if task_name not in anli:
            continue

        # if task_name not in tasks_indices_dict:
        #     tasks_indices_dict[task_name] = []

        group = "glue"
        if idx >= len(glue):                                    group = "super_glue"
        if idx >= len(glue) + len(superglue):                   group = "stanford"
        if idx >= len(glue) + len(superglue) + len(stanford):   group = "anli"
        labels = []

        if task_name == "rte":
            shuffle_key = shuffle_rules[task_name + "_g"] if group == "glue" else shuffle_rules[task_name + "_sg"]
        else:
            shuffle_key = shuffle_rules[task_name]

        dev_set = {"original": [], "shuffled": [], "shuffled_bigram": [], "shuffled_trigram": [], "swapped_nouns": []}
        swapped_stats[task_name] = {"swap_2_propNouns": {}, "swap_2_nouns": {}, "swap_2_nounPhrases": {}, "unchanged": {}}

        # Constructed a list of new examples
        if task_name in anli:
            new_examples = []
            input_dir = "datasets/smfa_nli/anli/"
            file_name = "dev" #"test"

            if task_name == "anli":
                file_path = input_dir + "full_unfiltered/" + file_name + ".txt"
            elif task_name == "r1":
                file_path = input_dir + "r1/" + file_name + ".jsonl"
            elif task_name == "r2":
                file_path = input_dir + "r2/" + file_name + ".jsonl"
            elif task_name == "r3":
                file_path = input_dir + "r3/" + file_name + ".jsonl"

            input_file = open(file_path, "r")
            for line in input_file:
                example = ast.literal_eval(line.strip())
                new_examples.append(example)
            input_file.close()
        else:
            # if task_name not in anli:
            #     continue

            print("********** " + task_name + " **********")
            dataset = nlp.load_dataset(task_name) if group == "stanford" else nlp.load_dataset(group, task_name)
            # dataset = dataset['train']
            dataset = dataset['validation'] if 'validation' in dataset else dataset['val']
            examples = dataset[:]
            assert shuffle_key in examples.keys()

            # ThangPM's NOTE 07-30-20
            # Only these tasks need to convert labels from integers to text
            if task_name in ["mnli_matched", "mnli_mismatched", "qnli", "rte", "snli"]:
                labels = dataset.features['label'].names

            if task_name == "squad_v2":
                print("Number of negative examples of SQuADv2: " + str(sum([len(example['text']) == 0 for example in examples['answers']])))

            # ThangPM's NOTE 07-28-20
            # Convert a dictionary (examples) with lists to a list of
            # dictionaries (new_examples) corresponding to each example.
            new_examples = []
            all_values = [examples[key] for key in examples.keys()]
            for idx in range(len(all_values[0])):
                input_dict = {}
                for idx2, key in enumerate(examples.keys()):
                    input_dict[key] = all_values[idx2][idx]
                new_examples.append(input_dict)

        for idx2, example in tqdm(enumerate(new_examples)):
        # for idx2, example in enumerate(new_examples):

            task_keys = get_task_keys((task_name + "_g") if task_name == "rte" else task_name)
            exp_label = new_examples[idx2][task_keys["label"]]
            exp_guid = new_examples[idx2][task_keys["guid"]]

            if task_name in tasks_indices_dict and exp_guid not in tasks_indices_dict[task_name]:
                continue

            # ***************** SHUFFLE EXAMPLES HERE *****************
            # ThangPM's NOTE 07-27-20
            # SuperGLUE and SQuAD will be handled in their own projects (jiant and finetuning)
            if task_name in glue + ["snli"] + ["squad_v2"] + anli:

                sentence_to_be_shuffled = example[shuffle_key]
                is_multi_sent_input, sentences, skip_removal = process_sentence_input(task_name, sentence_to_be_shuffled, idx2)
                sentence_to_be_shuffled = sentences[0]

                if is_multi_sent_input or len(sentence_to_be_shuffled.split(" ")) <= 3:
                    continue

                shuffled_unigram_sentence = shuffle_text(sentence_to_be_shuffled, type=ShuffleType.UNIGRAM, keep_punctuation=True, seed=seed)
                shuffled_bigram_sentence = shuffle_text(sentence_to_be_shuffled, type=ShuffleType.BIGRAM, keep_punctuation=True, seed=seed)
                shuffled_trigram_sentence = shuffle_text(sentence_to_be_shuffled, type=ShuffleType.TRIGRAM, keep_punctuation=True, seed=seed)

                # Swap Noun phrases:
                if two_noun_swap:
                    swapped_sentence, swapped = swap_two_nouns(sentence_to_be_shuffled, keep_punctuation=True, seed=seed, print_log=False)

                    if task_name == "stsb": exp_label = "swapped" if swapped else "original"

                    if swapped > 0:
                        if swapped == 1:
                            if exp_label not in swapped_stats[task_name]["swap_2_propNouns"]:
                                swapped_stats[task_name]["swap_2_propNouns"][exp_label] = 0
                            swapped_stats[task_name]["swap_2_propNouns"][exp_label] += 1
                        elif swapped == 2:
                            if exp_label not in swapped_stats[task_name]["swap_2_nouns"]:
                                swapped_stats[task_name]["swap_2_nouns"][exp_label] = 0
                            swapped_stats[task_name]["swap_2_nouns"][exp_label] += 1
                        elif swapped == 3:
                            if exp_label not in swapped_stats[task_name]["swap_2_nouns"]:
                                swapped_stats[task_name]["swap_2_nouns"][exp_label] = 0
                            swapped_stats[task_name]["swap_2_nouns"][exp_label] += 1
                    else:
                        if exp_label not in swapped_stats[task_name]["unchanged"]:
                            swapped_stats[task_name]["unchanged"][exp_label] = 0
                        swapped_stats[task_name]["unchanged"][exp_label] += 1

                    original_sent, swapped_sent = generate_input_example(new_examples[idx2], shuffle_key, swapped_sentence, task_name, group, labels)
                    dev_set["swapped_nouns"].append(swapped_sent)
                else:
                    dev_set["swapped_nouns"].append(None)

                original_sent, shuffled_unigram_sent = generate_input_example(new_examples[idx2], shuffle_key, shuffled_unigram_sentence, task_name, group, labels)
                original_sent, shuffled_bigram_sent = generate_input_example(new_examples[idx2], shuffle_key, shuffled_bigram_sentence, task_name, group, labels)
                original_sent, shuffled_trigram_sent = generate_input_example(new_examples[idx2], shuffle_key, shuffled_trigram_sentence, task_name, group, labels)

                dev_set["original"].append(original_sent)
                dev_set["shuffled"].append(shuffled_unigram_sent)
                dev_set["shuffled_bigram"].append(shuffled_bigram_sent)
                dev_set["shuffled_trigram"].append(shuffled_trigram_sent)

        if group != "super_glue":
            dev_sets[task_name] = dev_set

    # Write input examples to files for GLUE + SNLI
    if generate_output:

        for task_name, dev_set in dev_sets.items():
            original_examples, shuffled_examples, shuffled_bigram_examples, \
            shuffled_trigram_examples, swapped_nouns_examples = dev_set["original"], dev_set["shuffled"], dev_set["shuffled_bigram"], \
                                                                dev_set["shuffled_trigram"], dev_set["swapped_nouns"]

            if len(original_examples) == 0:
                continue

            # Rename these tasks to match transformer_gp2 project
            file_name = task_name
            if task_name == "mnli_matched":         file_name = "mnli"
            elif task_name == "mnli_mismatched":    file_name = "mnli-mm"
            elif task_name == "stsb":               file_name = "sts-b"
            elif task_name == "sst2":               file_name = "sst-2"

            data_dir_original = base_dir + str(seed) + "/dev_original/"
            data_dir_shuffled = base_dir + str(seed) + "/dev_shuffled/"
            data_dir_shuffled_bigram = base_dir + str(seed) + "/dev_shuffled_bigram/"
            data_dir_shuffle_trigram = base_dir + str(seed) + "/dev_shuffled_trigram/"
            data_dir_swapped_nouns = ""
            if two_noun_swap:
                data_dir_swapped_nouns = base_dir + str(seed) + "/dev_swapped_nouns/"

            if not exists(data_dir_original): makedirs(data_dir_original)
            if not exists(data_dir_shuffled): makedirs(data_dir_shuffled)
            if not exists(data_dir_shuffled_bigram): makedirs(data_dir_shuffled_bigram)
            if not exists(data_dir_shuffle_trigram): makedirs(data_dir_shuffle_trigram)
            if not exists(data_dir_swapped_nouns) and two_noun_swap: makedirs(data_dir_swapped_nouns)

            original_output = open(data_dir_original + file_name + ".txt", "w")
            shuffled_output = open(data_dir_shuffled + file_name + ".txt", "w")
            shuffled_bigram_output = open(data_dir_shuffled_bigram + file_name + ".txt", "w")
            shuffled_trigram_output = open(data_dir_shuffle_trigram + file_name + ".txt", "w")
            if two_noun_swap:
                swapped_nouns_output = open(data_dir_swapped_nouns + file_name + ".txt", "w")

            for original_example, shuffled_example, shuffled_bigram_example, shuffled_trigram_example, swapped_nouns_example in \
                    zip(original_examples, shuffled_examples, shuffled_bigram_examples, shuffled_trigram_examples, swapped_nouns_examples):

                if (task_name not in tasks_indices_dict) or (len(tasks_indices_dict[task_name]) > 0 and original_example.guid in tasks_indices_dict[task_name]):

                    if isinstance(original_example, dict):
                        original_output.write(str(original_example) + "\n")
                        shuffled_output.write(str(shuffled_example) + "\n")
                        shuffled_bigram_output.write(str(shuffled_bigram_example) + "\n")
                        shuffled_trigram_output.write(str(shuffled_trigram_example) + "\n")
                        if two_noun_swap:
                            swapped_nouns_output.write(str(swapped_nouns_example) + "\n")
                    else:
                        original_output.write(json.dumps(original_example.to_json()) + "\n")
                        shuffled_output.write(json.dumps(shuffled_example.to_json()) + "\n")
                        shuffled_bigram_output.write(json.dumps(shuffled_bigram_example.to_json()) + "\n")
                        shuffled_trigram_output.write(json.dumps(shuffled_trigram_example.to_json()) + "\n")
                        if two_noun_swap:
                            swapped_nouns_output.write(json.dumps(swapped_nouns_example.to_json()) + "\n")

            original_output.close()
            shuffled_output.close()
            shuffled_bigram_output.close()
            shuffled_trigram_output.close()
            if two_noun_swap:
                swapped_nouns_output.close()

    print("SWAPPED 2 NOUNS STATS")
    # if "stsb" in swapped_stats:
    #     swapped_stats.pop("stsb")
    print(swapped_stats)


def generate_synthetic_datasets(file_path, mode):
    import ast

    glue = ['sst-2', 'mrpc', 'qqp', 'qnli', 'rte', "sts-b", "squad_v2", "anli"]
    shuffle_attribute_dict = {"cola": "text_a", "mrpc": "text_a",
                              "qnli": "text_a", "qqp": "text_a",
                              "rte": "text_b", "sst-2": "text_a",
                              "sts-b": "text_a", "squad_v2": "question",
                              "anli": "text_b"}

    for task_name in glue:

        # if task_name != "squad_v2":
        #     continue

        if task_name != "anli":
            continue

        shuffle_key = shuffle_attribute_dict[task_name]
        file_name = task_name + ".txt"
        input_file = open(file_path + file_name, "r")
        output_file = open("preprocessed/synthetic/" + mode + "/" + file_name.replace(".txt", "_synthetic.txt"), "w")

        for idx, line in enumerate(input_file):
            # example = json.loads(line.strip())
            example = ast.literal_eval(line.strip())
            example["guid"] = idx
            example["text_a"] = example[shuffle_key]
            example["text_b"] = ""
            example["label"] = "1" # acceptable

            if task_name == "squad_v2":
                example.pop('id', None)
                example.pop('title', None)
                example.pop('context', None)
                example.pop('question', None)
                example.pop('answers', None)

            example_synthetic = copy.deepcopy(example)
            example_synthetic["text_a"] = shuffle_text(example_synthetic["text_a"], type=ShuffleType.ONE_WORD, keep_punctuation=True)
            example_synthetic["label"] = "0" # unacceptable

            output_file.write(json.dumps(example) + "\n")
            output_file.write(json.dumps(example_synthetic) + "\n")

        input_file.close()
        output_file.close()


def get_datasets_stats():
    glue = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli']
    superglue = ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc']
    stanford = ['snli', 'squad', 'squad_v2']

    all_sent_dicts, get_back_dicts = {}, {}
    dev_sets = {}
    short_sent_dict = {}

    # Used for ONLY 2 tasks: MultiRC and ReCoRD
    removed_indices = {'multirc': [], 'record': []}

    for idx, task_name in enumerate(glue + superglue + stanford):

        if task_name not in ["cola", "rte", "mrpc", "sst2", "qqp", "qnli"]:
            continue

        group = "glue"
        if idx >= len(glue):
            group = "super_glue"
        if idx >= len(glue) + len(superglue):
            group = "stanford"

        print("********** " + task_name + " **********")
        dataset = nlp.load_dataset(task_name) if group == "stanford" else nlp.load_dataset(group, task_name)
        dataset = dataset['validation'] if 'validation' in dataset else dataset['val']
        examples = dataset[:]

        labels = dataset.features['label'].names
        print(labels)

        label_dict = {}
        for label_id in examples["label"]:
            label_name = labels[label_id]
            if label_name not in label_dict:
                label_dict[label_name] = 0

            label_dict[label_name] += 1

        print(label_dict)


def merge_nli_datasets(input_dir, output_dir, search_name, combine=True):
    def unserialize_JsonableObject(d):
        global registered_jsonabl_classes
        classname = d.pop('_jcls_', None)
        if classname:
            cls = registered_jsonabl_classes[classname]
            obj = cls.__new__(cls)  # Make instance without calling __init__
            for key, value in d.items():
                setattr(obj, key, value)
            return obj
        else:
            return d

    files = glob.glob(input_dir + "/**/" + search_name, recursive=True)
    files = sorted(files, key=str.casefold)

    label_dict = {"e": "entailment", "n": "neutral", "c": "contradiction"}
    d_list = []

    if not exists(output_dir):
        makedirs(output_dir)

    for input_file in files:
        if not combine:
            d_list = []

        with open(input_file, encoding='utf-8', mode='r') as in_f:
            print("Load Jsonl:", input_file)
            for line in tqdm(in_f):
                item = json.loads(line.strip(), object_hook=unserialize_JsonableObject)

                # ThangPM: Convert data format of ANLI paper to Huggingface.
                example = InputExample(text_a=item["premise"], text_b=item["hypothesis"], label=label_dict[item["label"]], guid=item["uid"])
                d_list.append(example)

        if not combine:
            print(len(d_list))
            file_name = input_file.split("/")[-1].split(".")[0] + ".txt"
            output_file = open(output_dir + file_name, "w")
            [output_file.write(str(item.to_json()) + "\n") for item in d_list]
            output_file.close()

    if combine:
        print(len(d_list))
        file_name = input_file.split("/")[-1].split(".")[0] + ".txt"
        output_file = open(output_dir + file_name, "w")
        [output_file.write(str(item.to_json()) + "\n") for item in d_list]
        output_file.close()


# test_nlp()
# test_swag(Split.dev)
# test_semeval_and_atis(Split.dev, generate_output=True)

# print_example_glue(n_examples=5)
# print_example_superglue(n_examples=5)
# print_remaining_datasets(n_examples=5)

# get_datasets_stats()

# For 5 tasks in miniQ1 ONLY
# generate_synthetic_datasets(file_path="preprocessed/original/train/train_original/", mode="train")
# generate_synthetic_datasets(file_path="preprocessed/original/dev/dev_original/", mode="dev")

# file_name = "list_tasks_indices_anli_with_2_nouns_roberta.txt"
# two_noun_swap = True

# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=42, file_name="list_tasks_indices_adv_glue_roberta.txt", two_noun_swap=two_noun_swap)
# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=100, file_name="list_tasks_indices_adv_glue_bert.txt", two_noun_swap=two_noun_swap)
# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=200, file_name="list_tasks_indices_adv_glue_albert.txt", two_noun_swap=two_noun_swap)
# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=300, file_name="list_tasks_indices_adv_glue_roberta_large.txt", two_noun_swap=two_noun_swap)

# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=400, file_name="list_tasks_indices_extra_finetuning_1st.txt")
# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=500, file_name="list_tasks_indices_extra_finetuning_2nd.txt")
# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=600, file_name="list_tasks_indices_extra_finetuning_3rd.txt")
# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=700, file_name="list_tasks_indices_extra_finetuning_4th.txt")
# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=800, file_name="list_tasks_indices_extra_finetuning_5th.txt")

# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=42, file_name=file_name, two_noun_swap=two_noun_swap)
# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=100, file_name=file_name, two_noun_swap=two_noun_swap)
# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=200, file_name=file_name, two_noun_swap=two_noun_swap)
# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=300, file_name=file_name, two_noun_swap=two_noun_swap)
# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=400, file_name=file_name, two_noun_swap=two_noun_swap)
# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=500, file_name=file_name, two_noun_swap=two_noun_swap)
# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=600, file_name=file_name, two_noun_swap=two_noun_swap)
# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=700, file_name=file_name, two_noun_swap=two_noun_swap)
# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=800, file_name=file_name, two_noun_swap=two_noun_swap)
# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=900, file_name=file_name, two_noun_swap=two_noun_swap)
# general_all_shuffled_datasets_from_files(generate_output=True, base_dir="preprocessed/", seed=1000, file_name=file_name, two_noun_swap=two_noun_swap)

# general_all_shuffled_datasets_from_scratch(generate_output=True, base_dir="preprocessed/", seed=42)

general_all_shuffled_datasets_from_scratch(generate_output=True, base_dir="preprocessed/", seed=100)
general_all_shuffled_datasets_from_scratch(generate_output=True, base_dir="preprocessed/", seed=200)
general_all_shuffled_datasets_from_scratch(generate_output=True, base_dir="preprocessed/", seed=300)
general_all_shuffled_datasets_from_scratch(generate_output=True, base_dir="preprocessed/", seed=400)
general_all_shuffled_datasets_from_scratch(generate_output=True, base_dir="preprocessed/", seed=500)
general_all_shuffled_datasets_from_scratch(generate_output=True, base_dir="preprocessed/", seed=600)
general_all_shuffled_datasets_from_scratch(generate_output=True, base_dir="preprocessed/", seed=700)
general_all_shuffled_datasets_from_scratch(generate_output=True, base_dir="preprocessed/", seed=800)
general_all_shuffled_datasets_from_scratch(generate_output=True, base_dir="preprocessed/", seed=900)
general_all_shuffled_datasets_from_scratch(generate_output=True, base_dir="preprocessed/", seed=1000)

# print_out_task_keys()

# Merge NLI datasets (SNLI, MNLI, FEVER, ANLI) to form a combination for finetuning.
# merge_nli_datasets(input_dir="datasets/smfa_nli/", output_dir="preprocessed/smfa_nli/", search_name="train.jsonl")
# merge_nli_datasets(input_dir="datasets/smfa_nli/mnli/", output_dir="preprocessed/smfa_nli/", search_name="*dev.jsonl", combine=False)

'''
*** How to generate synthetic ANLI ***
Step 1: Merge R1, R2 and R3 -> ANLI_full_unfiltered
Step 2: Processed ANLI_full_unfiltered to remove multi-sentence hypotheses or len(hypothesis) <= 3 -> original ANLI in preprocessed
Step 3: Generate synthetic datasets from original ANLI in preprocessed
'''
# ====================================================================================================================================
# STEP 1
# merge_nli_datasets(input_dir="datasets/smfa_nli/anli/", output_dir="preprocessed/anli/", search_name="train.jsonl")
# merge_nli_datasets(input_dir="datasets/smfa_nli/anli/", output_dir="preprocessed/anli/", search_name="*dev.jsonl")
# merge_nli_datasets(input_dir="datasets/smfa_nli/anli/", output_dir="preprocessed/anli/", search_name="*test.jsonl")

# STEP 2: run general_all_shuffled_datasets_from_scratch

# STEP 3
# generate_synthetic_datasets(file_path="preprocessed/original/train/train_original/", mode="train")
# generate_synthetic_datasets(file_path="preprocessed/original/dev/dev_original/", mode="dev")
# generate_synthetic_datasets(file_path="preprocessed/original/test/test_original/", mode="test")
# ====================================================================================================================================

