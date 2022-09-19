# import transformers
from transformers import BertConfig, BertTokenizer
# import textattack

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset

from textattack.datasets import HuggingFaceDataset
import json
from src.transformers.data.processors.utils import InputExample

# from thangpm.bert.atae_bert import ATAE_BERT
# from thangpm.glue_utils import ABSAProcessor

# MODEL_CLASSES = {
#     'bert': (BertConfig, ATAE_BERT, BertTokenizer)
# }

task_name = "RTE"
num_labels = 2
model_name_or_path = "../../examples/models/roberta-extra-finetune/3rd/RTE/finetuned/"
data_path = "../../examples/glue_data/val_new/miniQ1_filtered_by_len/dev_original/rte.txt"

''' SAMPLE
tokenizer = textattack.models.tokenizers.AutoTokenizer(model_path)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
dataset = textattack.datasets.HuggingFaceNLPDataset("glue", subset="sst2", split="train", shuffle=False)
'''

def get_dataset():
    # processor = ABSAProcessor()
    # label_list = processor.get_labels("SENT_LEVEL")
    # aspect_list = ['food', 'misc', 'service', 'ambience', 'price']
    # label_map = {label: i for i, label in enumerate(label_list)}

    label_list = ["entailment", "not_entailment"]
    label_map = {label: i for i, label in enumerate(label_list)}

    examples = []
    input_file = open(data_path, "r")
    for line in input_file:
        json_obj = json.loads(line.strip())
        example = InputExample(guid=json_obj['guid'],
                               text_a=json_obj['text_a'],
                               text_b=(json_obj['text_b'] if json_obj['text_b'] != "" else None),
                               # For RoBERTa tokenizer
                               label=json_obj['label'])
        examples.append(example)

    # examples = processor.get_test_examples(data_path, tagging_schema="SENT_LEVEL", token_level=False)

    dataset = []
    for example in examples:
        attack_example = {"text": example.text_a, "label": label_map[example.label]}
        dataset.append(attack_example)

    # Prepare Huggingface dataset
    hf_dataset = HuggingFaceDataset(task_name, subset=None, split="dev", label_map=None)
    hf_dataset.examples = dataset
    hf_dataset.input_columns = ("text",)
    hf_dataset.output_column = "label"
    hf_dataset.label_names = label_list

    return hf_dataset

# _, model_class, tokenizer_class = MODEL_CLASSES["bert"]
# model = model_class.from_pretrained(model_path)
# tokenizer = tokenizer_class.from_pretrained(model_path)

# Update tokenizer
# tokenizer.add_tokens(["misc", "ambience"])
# model.resize_token_embeddings(len(tokenizer))

config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    finetuning_task=task_name,
)
config.fix_tfm = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    from_tf=False,
    config=config,
)

dataset = get_dataset()


# def Attack(model):
#     goal_function = textattack.goal_functions.UntargetedClassification(model)
#     search_method = textattack.search_methods.GreedyWordSwapWIR()
#     transformation = textattack.transformations.WordSwapRandomCharacterSubstitution()
#     constraints = []
#
#     return textattack.shared.Attack(goal_function, constraints, transformation, search_method)


# attack = Attack(model)
# attack.attack_one()