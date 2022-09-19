
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import EvalPrediction, GlueDataset, InputExample, pipeline

import textattack
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper

import json

# task_names = ["rte", "mrpc", "qqp", "sst-2", "qnli"]
# recipes = ["textfooler", "input-reduction", "deepwordbug", "bae", "pwws"]

task_name = "RTE"
base_model = False

if base_model:
    model_name_or_path = "../models/roberta-base/" + task_name + "/finetuned/"
else:
    model_name_or_path = "../models/roberta-extra-finetune/3rd/" + task_name + "/finetuned/"

num_labels = 2
data_path = "../datasets/val_new/miniQ1_filtered_by_len/dev_original/" + task_name.lower() + ".txt"

''' SAMPLE
tokenizer = textattack.models.tokenizers.AutoTokenizer(model_path)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
dataset = textattack.datasets.HuggingFaceNLPDataset("glue", subset="sst2", split="train", shuffle=False)
'''

def get_dataset():
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

    dataset = []
    for example in examples:
        attack_example = {"text": example.text_a, "label": label_map[example.label]}
        dataset.append(attack_example)

    # Prepare Huggingface dataset
    hf_dataset = HuggingFaceDataset("glue", subset=task_name.lower(), split="validation", label_map=None)
    hf_dataset.examples = dataset
    hf_dataset.input_columns = ("text",)
    hf_dataset.output_column = "label"
    hf_dataset.label_names = label_list

    return hf_dataset


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

#pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
#model = HuggingFaceSentimentAnalysisPipelineWrapper(pipeline)
model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

#dataset = get_dataset()
dataset = textattack.datasets.HuggingFaceDataset("glue", subset=task_name, split="validation", shuffle=False)


