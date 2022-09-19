import json
import spacy


base_data_dir = "datasets/full"

task_groups = {"glue": ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst-2", "sts-b", "wnli"],
               "super-glue": ["boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc"],
               "stanford": ["squad_v1", "squad_v2", "snli"],
               "others": ["swag", "semeval", "atis", "atis_ner"]}

modes = ["original", "shuffled", "shuffled_bigram", "shuffled_trigram"]


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def print_out_examples(path_to_file, count=5):
    input_file = open(path_to_file, "r")
    for idx, line in enumerate(input_file):
        if idx < count:
            print(line)
        else:
            break

    input_file.close()


def get_examples(path_to_file, count=0):
    examples = []

    input_file = open(path_to_file, "r")
    for idx, line in enumerate(input_file):
        # Get full examples if count == 0
        if idx < count or count == 0:
            examples.append(line.strip())
        else:
            break

    input_file.close()
    return examples


def run_tester():
    for group, tasks in task_groups.items():
        for task in tasks:
            data_dir = "/".join([base_data_dir, group, task])

            print("********** " + group + " --- " + task + " **********")
            full_examples = []

            for mode in modes:
                ext = ".jsonl" if group == "super-glue" else ".txt"
                data_file = data_dir + ("_" + mode + ext if mode != "original" else ext)
                examples = get_examples(data_file, count=5)
                full_examples.append(examples)

            for original, shuffled, shuffled_bigram, shuffled_trigram in zip(*full_examples):
                print("-" * 30)
                print(color.BOLD + "Original: " + color.END + original)
                print(color.BOLD + "Shuffled: " + color.END + shuffled)
                print(color.BOLD + "Shuffled_bigram: " + color.END + shuffled_bigram)
                print(color.BOLD + "Shuffled_trigram: " + color.END + shuffled_trigram)
                print("-" * 30)

            print("\n")


def get_data_length_statistics():

    task_keys = {"boolq" : ['question', 'passage'],
                 "cb" : ['premise', 'hypothesis'],
                 "copa" : ['premise', 'choice1', 'choice2', 'question'],
                 "multirc" : ['paragraph', 'question', 'answer'],
                 "record" : ['passage', 'query', 'entities', 'answers'],
                 "rte_sg": ['premise', 'hypothesis'],
                 "wic" : ['sentence1', 'sentence2'],
                 "wsc" : ['text'],

                 "squad_v1" : ['context', 'question', 'answers'],
                 "squad_v2" : ['context', 'question', 'answers'],
                 "swag": ["contexts", "endings", "question"]}

    for group, tasks in task_groups.items():
        for task in tasks:
            data_dir = "/".join([base_data_dir, group, task])

            print("********** " + group + " --- " + task + " **********")

            ext = ".jsonl" if group == "super-glue" else ".txt"
            data_file = data_dir + ext
            examples = get_examples(data_file)

            stats = {"min_len": 100, "max_len": 0, "avg_len": 0}
            diff_len = []
            total_len = 0

            for example in examples:
                # Will handle these tasks later
                if task == "multirc" or task == "record":
                    continue

                obj = json.loads(example)

                if task == "boolq":
                    input_len = len(obj["question"].split(" ")) + len(obj["passage"].split(" "))
                elif task == "cb":
                    input_len = len(obj["premise"].split(" ")) + len(obj["hypothesis"].split(" "))
                elif task == "copa":
                    len1 = len(obj["premise"].split(" ")) + len(obj["choice1"].split(" "))
                    len2 = len(obj["premise"].split(" ")) + len(obj["choice2"].split(" "))
                    input_len = max(len1, len2)
                elif task == "multirc":
                    input_len = len(obj["paragraph"].split(" ")) + len(obj["question"].split(" ")) # Update later
                elif task == "record":
                    input_len = len(obj["passage"].split(" ")) + len(obj["query"].split(" ")) # Update later
                elif task == "rte_sg":
                    input_len = len(obj["premise"].split(" ")) + len(obj["hypothesis"].split(" "))
                elif task == "wic":
                    input_len = len(obj["sentence1"].split(" ")) + len(obj["sentence2"].split(" "))
                elif task == "wsc":
                    input_len = len(obj["text"].split(" "))
                elif task == "squad_v1":
                    input_len = len(obj["context_text"].split(" ")) + len(obj["question_text"].split(" "))
                elif task == "squad_v2":
                    input_len = len(obj["context_text"].split(" ")) + len(obj["question_text"].split(" "))
                elif task == "swag":
                    len1 = len(obj["contexts"][0].split(" ")) + len(obj["endings"][0].split(" "))
                    len2 = len(obj["contexts"][0].split(" ")) + len(obj["endings"][1].split(" "))
                    len3 = len(obj["contexts"][0].split(" ")) + len(obj["endings"][2].split(" "))
                    len4 = len(obj["contexts"][0].split(" ")) + len(obj["endings"][2].split(" "))
                    input_len = max(len1, len2, len3, len4)
                elif task == "rte" and group == "super-glue":
                    input_len = len(obj["premise"].split(" ")) + len(obj["hypothesis"].split(" "))
                elif task == "qqp":
                    diff_len.append(len(obj["text_a"].split(" ")) - len(obj["text_b"].split(" ")))
                else:
                    input_len = len(obj["text_a"].split(" ")) + (len(obj["text_b"].split(" ")) if "text_b" in obj and obj["text_b"] != None else 0)

                if input_len > stats["max_len"]: stats["max_len"] = input_len
                elif input_len < stats["min_len"]: stats["min_len"] = input_len

                total_len += input_len

            stats["avg_len"] = total_len / len(examples)
            if len(diff_len) != 0:
                stats["avg_diff_len"] = sum(diff_len) / len(diff_len)
                print("Max Diff Len = " + str(max(diff_len)))
                print("Min Diff Len = " + str(min(diff_len)))
                print(diff_len)
                print(len(diff_len))

            print(str(stats) + "\n")


def swap_noun_phrases(text):
    nlp_spacy = spacy.load("en_core_web_sm")

    doc = nlp_spacy(text)

    # Analyze syntax
    if len(list(doc.noun_chunks)) > 0:
        print("--------------------------------------------------------------------")
        print(text)
        print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
        print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

        # Find named entities, phrases and concepts
        for entity in doc.ents:
            print(entity.text, entity.label_)

        merge_nps = nlp_spacy.create_pipe("merge_noun_chunks")
        nlp_spacy.add_pipe(merge_nps)

        texts = [t.text for t in nlp_spacy(text)]
        print(texts)


def print_logs_adv_glue_stats():

    stats_dict = {
        "roberta": {'cola': {'swap_2_propNouns': {0: 31, 1: 26}, 'swap_2_nouns': {1: 78, 0: 68}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 50, 0: 55}}, 'sst2': {'swap_2_propNouns': {0: 18, 1: 34}, 'swap_2_nouns': {1: 318, 0: 316}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 50, 0: 68}}, 'mrpc': {'swap_2_propNouns': {0: 49, 1: 36}, 'swap_2_nouns': {1: 63, 0: 51}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 2, 0: 1}}, 'qqp': {'swap_2_propNouns': {1: 1205, 0: 1657}, 'swap_2_nouns': {0: 7868, 1: 7078}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 3158, 1: 4400}}, 'qnli': {'swap_2_propNouns': {0: 360, 1: 388}, 'swap_2_nouns': {1: 1657, 0: 1683}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 455, 0: 457}}, 'rte': {'swap_2_propNouns': {0: 36, 1: 21}, 'swap_2_nouns': {1: 47, 0: 30}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 6, 1: 4}}},
        "bert": {'cola': {'swap_2_propNouns': {1: 27, 0: 29}, 'swap_2_nouns': {1: 82, 0: 70}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 48, 0: 58}}, 'sst2': {'swap_2_propNouns': {0: 17, 1: 32}, 'swap_2_nouns': {1: 311, 0: 312}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 49, 0: 63}}, 'mrpc': {'swap_2_propNouns': {0: 47, 1: 28}, 'swap_2_nouns': {0: 43, 1: 60}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 2}}, 'qqp': {'swap_2_propNouns': {1: 1168, 0: 1624}, 'swap_2_nouns': {0: 7704, 1: 6927}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 3065, 1: 4298}}, 'qnli': {'swap_2_propNouns': {0: 362, 1: 390}, 'swap_2_nouns': {0: 1671, 1: 1647}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 437, 0: 441}}, 'rte': {'swap_2_propNouns': {0: 31, 1: 24}, 'swap_2_nouns': {1: 45, 0: 34}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 8, 1: 4}}},
        "albert": {'cola': {'swap_2_propNouns': {1: 30, 0: 38}, 'swap_2_nouns': {1: 95, 0: 74}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 64, 1: 51}}, 'sst2': {'swap_2_propNouns': {0: 19, 1: 36}, 'swap_2_nouns': {1: 309, 0: 309}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 47, 0: 64}}, 'mrpc': {'swap_2_propNouns': {0: 47, 1: 33}, 'swap_2_nouns': {0: 51, 1: 64}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 2, 0: 1}}, 'qqp': {'swap_2_propNouns': {1: 1173, 0: 1619}, 'swap_2_nouns': {0: 7651, 1: 6949}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 3085, 1: 4233}}, 'qnli': {'swap_2_propNouns': {0: 369, 1: 385}, 'swap_2_nouns': {0: 1676, 1: 1656}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 447, 0: 443}}, 'rte': {'swap_2_propNouns': {1: 28, 0: 41}, 'swap_2_nouns': {1: 58, 0: 42}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 9, 1: 6}}},
        "roberta-large": {'cola': {'swap_2_propNouns': {1: 31, 0: 35}, 'swap_2_nouns': {1: 82, 0: 78}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 65, 0: 65}}, 'sst2': {'swap_2_propNouns': {0: 18, 1: 34}, 'swap_2_nouns': {1: 322, 0: 323}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 53, 0: 68}}, 'mrpc': {'swap_2_propNouns': {0: 48, 1: 47}, 'swap_2_nouns': {1: 54, 0: 55}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 2}}, 'qqp': {'swap_2_propNouns': {1: 1209, 0: 1653}, 'swap_2_nouns': {0: 7897, 1: 7138}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 3165, 1: 4368}}, 'qnli': {'swap_2_propNouns': {0: 377, 1: 395}, 'swap_2_nouns': {1: 1704, 0: 1713}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 454, 0: 463}}, 'rte': {'swap_2_propNouns': {1: 32, 0: 38}, 'swap_2_nouns': {1: 59, 0: 50}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 8, 1: 5}}},
        "roberta-extra-finetuned-1st": {'cola': {'swap_2_propNouns': {}, 'swap_2_nouns': {}, 'swap_2_nounPhrases': {}, 'unchanged': {}}, 'sst2': {'swap_2_propNouns': {0: 18, 1: 32}, 'swap_2_nouns': {1: 315, 0: 313}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 50, 0: 66}}, 'mrpc': {'swap_2_propNouns': {0: 46, 1: 30}, 'swap_2_nouns': {1: 61, 0: 46}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 1}}, 'qqp': {'swap_2_propNouns': {1: 1197, 0: 1684}, 'swap_2_nouns': {0: 7806, 1: 7085}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 3160, 1: 4368}}, 'qnli': {'swap_2_propNouns': {0: 365, 1: 393}, 'swap_2_nouns': {0: 1688, 1: 1670}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 446, 0: 456}}, 'rte': {'swap_2_propNouns': {0: 34, 1: 24}, 'swap_2_nouns': {1: 46, 0: 35}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 5, 1: 4}}},
        "roberta-extra-finetuned-2nd": {'cola': {'swap_2_propNouns': {}, 'swap_2_nouns': {}, 'swap_2_nounPhrases': {}, 'unchanged': {}}, 'sst2': {'swap_2_propNouns': {0: 18, 1: 34}, 'swap_2_nouns': {1: 312, 0: 311}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 66, 1: 49}}, 'mrpc': {'swap_2_propNouns': {0: 44, 1: 34}, 'swap_2_nouns': {1: 50, 0: 41}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 1}}, 'qqp': {'swap_2_propNouns': {1: 1201, 0: 1687}, 'swap_2_nouns': {0: 7860, 1: 7132}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 3153, 1: 4367}}, 'qnli': {'swap_2_propNouns': {0: 365, 1: 389}, 'swap_2_nouns': {0: 1693, 1: 1675}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 446, 0: 452}}, 'rte': {'swap_2_propNouns': {1: 24, 0: 34}, 'swap_2_nouns': {1: 47, 0: 36}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 5, 1: 4}}},
        "roberta-extra-finetuned-3rd": {'cola': {'swap_2_propNouns': {}, 'swap_2_nouns': {}, 'swap_2_nounPhrases': {}, 'unchanged': {}}, 'sst2': {'swap_2_propNouns': {0: 18, 1: 32}, 'swap_2_nouns': {1: 318, 0: 315}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 50, 0: 67}}, 'mrpc': {'swap_2_propNouns': {0: 46, 1: 38}, 'swap_2_nouns': {1: 59, 0: 52}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 1}}, 'qqp': {'swap_2_propNouns': {1: 1203, 0: 1675}, 'swap_2_nouns': {0: 7955, 1: 7118}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 3093, 1: 4402}}, 'qnli': {'swap_2_propNouns': {0: 366, 1: 387}, 'swap_2_nouns': {0: 1693, 1: 1680}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 448, 0: 456}}, 'rte': {'swap_2_propNouns': {0: 34, 1: 25}, 'swap_2_nouns': {1: 50, 0: 40}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 5, 1: 4}}},
        "roberta-extra-finetuned-4th": {'cola': {'swap_2_propNouns': {}, 'swap_2_nouns': {}, 'swap_2_nounPhrases': {}, 'unchanged': {}}, 'sst2': {'swap_2_propNouns': {0: 18, 1: 32}, 'swap_2_nouns': {1: 315, 0: 314}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 52, 0: 67}}, 'mrpc': {'swap_2_propNouns': {0: 42, 1: 38}, 'swap_2_nouns': {0: 46, 1: 49}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 2, 0: 1}}, 'qqp': {'swap_2_propNouns': {1: 1207, 0: 1670}, 'swap_2_nouns': {0: 7939, 1: 7146}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 3138, 1: 4394}}, 'qnli': {'swap_2_propNouns': {0: 370, 1: 391}, 'swap_2_nouns': {1: 1674, 0: 1689}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 447, 0: 453}}, 'rte': {'swap_2_propNouns': {0: 31, 1: 21}, 'swap_2_nouns': {1: 44, 0: 34}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 4, 1: 4}}},
        "roberta-extra-finetuned-5th": {'cola': {'swap_2_propNouns': {}, 'swap_2_nouns': {}, 'swap_2_nounPhrases': {}, 'unchanged': {}}, 'sst2': {'swap_2_propNouns': {0: 18, 1: 34}, 'swap_2_nouns': {1: 317, 0: 314}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 67, 1: 48}}, 'mrpc': {'swap_2_propNouns': {0: 49, 1: 38}, 'swap_2_nouns': {1: 62, 0: 51}, 'swap_2_nounPhrases': {}, 'unchanged': {}}, 'qqp': {'swap_2_propNouns': {1: 1205, 0: 1678}, 'swap_2_nouns': {0: 7846, 1: 7074}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 4373, 0: 3128}}, 'qnli': {'swap_2_propNouns': {0: 364, 1: 397}, 'swap_2_nouns': {0: 1692, 1: 1665}, 'swap_2_nounPhrases': {}, 'unchanged': {1: 448, 0: 454}}, 'rte': {'swap_2_propNouns': {0: 31, 1: 23}, 'swap_2_nouns': {1: 46, 0: 37}, 'swap_2_nounPhrases': {}, 'unchanged': {0: 5, 1: 4}}},
    }
    task_names = ["cola", "rte", "qqp", "sst2", "mrpc", "qnli"]

    for model, stats in stats_dict.items():
        print("***** STATS of model: " + model + " *****")

        for task_name in task_names:
            pos_stats, neg_stats = [], []
            # print("Task: " + task_name)

            for key, values in stats[task_name].items():
                if len(values) == 0:
                    neg_stats.append(0)
                    pos_stats.append(0)
                    continue

                if task_name == "rte" or task_name == "qnli":
                    pos_stats.append(values[0] if 0 in values else 0)
                    neg_stats.append(values[1] if 1 in values else 0)
                else:
                    neg_stats.append(values[0] if 0 in values else 0)
                    pos_stats.append(values[1] if 1 in values else 0)

            print("\t".join([str(x) for x in neg_stats]))
            print("\t".join([str(x) for x in pos_stats]))


if __name__ == '__main__':
    # run_tester()
    # get_data_length_statistics()

    # swap_noun_phrases("The jeweller scribbled the contract with his name.")
    # swap_noun_phrases(" ".join(['How', 'can', 'I', 'expand', 'my', 'IQ?']))
    # print_logs_adv_glue_stats()

    '''
        "The jeweller scribbled the contract with his name."
        [0,     1,       2,      3,     4,     5,  6,   7]
        
    ->  "the contract scribbled The jeweller with his name."
        [3,     4,       2,      0,     1,     5,   6,  7]
    '''