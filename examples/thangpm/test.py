from torch.utils.tensorboard import SummaryWriter
import numpy as np

from os import makedirs, mkdir, listdir
from os.path import isfile, join, exists

import statistics

import pickle
import matplotlib.pyplot as plt


def demo_tensorboard():
    writer = SummaryWriter()

    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)


def get_multiple_runs_results():
    # base_dir = "../models/bert-base-uncased/"
    # tasks = ["MRPC", "RTE"]

    # base_dir = "../models/roberta-base/"
    # base_dir = "../models/albert-base-v2/"
    base_dir = "../models/bert-base-uncased/"
    # base_dir = "../models/roberta-large/"

    # base_dir = "../models/roberta-extra-finetune/"

    # tasks = ["CoLA", "MRPC", "SST-2", "RTE", "QQP", "QNLI", "STS-B"]
    tasks = ["MRPC", "QQP"]
    synthetic_tasks = [""]

    # tasks = ["MNLI"]
    # synthetic_tasks = ["r1", "r2", "r3"]

    # seeds = [422]
    # seeds = [42, 100, 200, 300, 400]
    # modes = ["swapped_nouns"]

    seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # modes = ["original", "shuffled", "shuffled_bigram", "shuffled_trigram"]
    modes = ["shuffled_2_sents"]

    results = {}

    for task in tasks:
        # if task == "CoLA" or task == "STS-B":
        #     continue

        results[task] = {}

        for seed in seeds:
            for synthetic_task in synthetic_tasks:
                if synthetic_task not in results[task]:
                    results[task][synthetic_task] = {}

                for mode in modes:
                    print("TASK: {} - SEED: {} - MODE: {}".format(task, seed, mode))
                    # print("TASK: {} - SEED: {} - SUB_TASK: {} - MODE: {}".format(task, seed, synthetic_task, mode))

                    result_idx = 1 if task != "STS-B" else 2 # Spearmanr for STS-B
                    if mode not in results[task][synthetic_task]:
                        results[task][synthetic_task][mode] = {}

                    result_dir = base_dir + task + "/evaluation_rebuttal/" + str(seed) + "/" + synthetic_task + "/" + mode + "/"  # Normal finetuning
                    # result_dir = base_dir + str(seed) + "/" + task + "/evaluation/" + synthetic_task + "/" + mode + "/"  # Extra finetuning
                    result_fp = result_dir + "eval_results_" + task.lower() + ".txt"
                    result_file = open(result_fp, "r")
                    lines = [line for line in result_file]
                    print(lines[result_idx])

                    key, score = lines[result_idx].split("=")
                    key, score = key.strip(), score.strip()

                    if key not in results[task][synthetic_task][mode]:
                        results[task][synthetic_task][mode][key] = []

                    results[task][synthetic_task][mode][key].append(str(round(float(score) * 100, 2)))

    # print(results)

    for task, synthetic_tasks in results.items():
        print("***** TASK NAME: " + task + " *****")

        for synthetic_task, shuffle_types in synthetic_tasks.items():

            for shuffle, values in shuffle_types.items():
                print("+++ Shuffle Type: " + shuffle + " +++")
                key = "eval_acc" if task != "STS-B" else "eval_spearmanr" #"eval_pearson"
                if task == "MNLI":
                    key = "eval_mnli/acc"

                print("\t".join(values[key]) + " || " + str(round(statistics.mean([float(value) for value in values[key]]), 2)))


def get_results():
    # base_model = "bert-base-uncased"
    # base_model = "roberta-base"
    # base_model = "albert-base-v2"

    # base_dir = "../models/" + base_model + "/"
    # tasks = ["CoLA", "MNLI", "MRPC", "QNLI", "QQP", "RTE", "SST-2", "STS-B", "WNLI"]
    # modes = ["evaluation",] #  "finetuned", "evaluation"
    # shuffled_types = ["baseline", "original", "shuffled", "shuffled_bigram", "shuffled_trigram"]

    base_model = "roberta-extra-finetune"
    base_dir = "../models/" + base_model + "/"
    seeds = ["1st_42", "2nd_100", "3rd_200", "4th_300", "5th_400"]
    tasks = ["MNLI"]
    modes = ["evaluation", ]
    synthetic_tasks = ["r1", "r2", "r3"]
    shuffled_types = ["shuffled",]

    for task in tasks:
        for seed in seeds:
            for mode in modes:
                for synthetic_task in synthetic_tasks:
                    for type in shuffled_types:
                        # print("TASK: {} - MODE: {} - TYPE: {}".format(task, mode, type))
                        print("TASK: {} - MODE: {} - TYPE: {} -  SUB_TASK: {}".format(task, mode, type, synthetic_task))

                        result_dir = base_dir + seed + "/" + task + "/" + mode + "/" + synthetic_task + "/"
                        result_dir += (type + "/") if mode == "evaluation" else ""

                        result_fp = result_dir + "eval_results_" + task.lower() + ".txt"
                        if not exists(result_fp): continue

                        result_file = open(result_fp, "r")
                        lines = [line for line in result_file]
                        print(lines[1])

                        # if task == "MNLI":
                        #     print("TASK: {} - MODE: {} - TYPE: {}".format(task + "-MM", mode, type))
                        #     result_fp = result_dir + "eval_results_" + task.lower() + "-mm" + ".txt"
                        #     if not exists(result_fp): continue
                        #
                        #     result_file = open(result_fp, "r")
                        #     lines = [line for line in result_file]
                    #     print(lines[1])


def compute_calibration_error(bucket_size, task_name, time="1st"):

    # time = "2nd"
    # if task_name == "RTE": time = "1st"
    # elif task_name == "SST-2": time = "3rd"
    # elif task_name == "MRPC": time = "5th"

    ml_objects_path = "../models/roberta-base/" + task_name + "/evaluation-2nd/original/"
    ml_objects_extra_path = "../models/roberta-extra-finetune/" + time + "/" + task_name + "/evaluation_1st/original/"

    # print(ml_objects_path)
    # print(ml_objects_extra_path)

    with open(ml_objects_path + "ml_objects.pickle", "rb") as file_path:
        ml_objects = pickle.load(file_path)

    with open(ml_objects_extra_path + "ml_objects.pickle", "rb") as file_path:
        ml_objects_extra = pickle.load(file_path)

    calibration_error_original, calibration_error_extra = 0, 0
    confidence_range = 1 / bucket_size
    cal_dict_original, cal_dict_extra = {}, {}

    for i in range(bucket_size):
        cal_dict_original[round((i + 1) * confidence_range, 2)] = {"accuracy": [], "confidence": []}
        cal_dict_extra[round((i + 1) * confidence_range, 2)] = {"accuracy": [], "confidence": []}

    for obj, obj_extra in zip(ml_objects, ml_objects_extra):

        confidence_score_orginal = np.max(obj.get_confidence_score())
        confidence_score_extra = np.max(obj_extra.get_confidence_score())

        flag_original, flag_extra = True, True

        for key in sorted(cal_dict_original.keys()):

            if flag_original and confidence_score_orginal <= key:
                cal_dict_original[key]["confidence"].append(confidence_score_orginal)

                if obj.get_pred_label() == obj.get_ground_truth():
                    cal_dict_original[key]["accuracy"].append(1)
                else:
                    cal_dict_original[key]["accuracy"].append(0)

                flag_original = False

            if flag_extra and confidence_score_extra <= key:
                cal_dict_extra[key]["confidence"].append(confidence_score_extra)

                if obj_extra.get_pred_label() == obj_extra.get_ground_truth():
                    cal_dict_extra[key]["accuracy"].append(1)
                else:
                    cal_dict_extra[key]["accuracy"].append(0)

                flag_extra = False

            if not flag_original and not flag_extra:
                break

    accuracy_avg_list_original, confidence_avg_list_original = [], []
    accuracy_avg_list_extra, confidence_avg_list_extra = [], []

    for key in sorted(cal_dict_original.keys()):

        def compute_cal_error_per_bin(bucket_ith, n_samples):
            bucket_ith_count = len(bucket_ith["accuracy"])
            assert len(bucket_ith["accuracy"]) == len(bucket_ith["confidence"])

            if bucket_ith_count == 0:
                return 0, 0, 0

            accuracy_avg = sum(bucket_ith["accuracy"]) / bucket_ith_count
            confidence_avg = sum(bucket_ith["confidence"]) / bucket_ith_count
            cal_per_bin = abs(accuracy_avg - confidence_avg) * bucket_ith_count / n_samples

            return cal_per_bin, accuracy_avg, confidence_avg

        # Compute calibration error per bucket
        cal_per_bin_original, accuracy_avg_original, confidence_avg_original = compute_cal_error_per_bin(
            bucket_ith=cal_dict_original[key], n_samples=len(ml_objects))
        cal_per_bin_extra, accuracy_avg_extra, confidence_avg_extra = compute_cal_error_per_bin(
            bucket_ith=cal_dict_extra[key], n_samples=len(ml_objects_extra))

        # Sum all calibration errors from all buckets
        calibration_error_original += cal_per_bin_original
        calibration_error_extra += cal_per_bin_extra

        accuracy_avg_list_original.append(accuracy_avg_original)
        confidence_avg_list_original.append(confidence_avg_original)

        accuracy_avg_list_extra.append(accuracy_avg_extra)
        confidence_avg_list_extra.append(confidence_avg_extra)

    calibration_error_original = round(calibration_error_original * 100, 2)
    calibration_error_extra = round(calibration_error_extra * 100, 2)

    print("Task name: " + task_name)
    print("Calibration Error (Original): " + str(calibration_error_original))
    print("Calibration Error (Extra): " + str(calibration_error_extra))

    # plot_calibration_figure_type1(cal_dict_original, title="Expected Calibration Error - Normal Finetuning",
    #                               file_name="calibration_error_" + task_name + "_normal_finetuning.jpg")
    # plot_calibration_figure_type1(cal_dict_extra, title="Expected Calibration Error - Extra Finetuning",
    #                               file_name="calibration_error_" + task_name + "_extra_finetuning.jpg")
    #
    # plot_calibration_figure_type2(cal_dict_original, calibration_error_original,
    #                               title="Expected Calibration Error - Normal Finetuning",
    #                               file_name="calibration_error_" + task_name + "_normal_finetuning_type2.jpg")
    # plot_calibration_figure_type2(cal_dict_extra, calibration_error_extra,
    #                               title="Expected Calibration Error - Extra Finetuning",
    #                               file_name="calibration_error_" + task_name + "_extra_finetuning_type2.jpg")

    return calibration_error_original, calibration_error_extra

def plot_calibration_figure_type1(cal_dict, title, file_name):
    labels = []
    total_preds = []
    correct_preds = []
    confidence_scores = []

    max_examples_per_column = 0

    for key in sorted(cal_dict.keys()):
        labels.append(key)
        total_preds.append(len(cal_dict[key]["accuracy"]))
        correct_preds.append(sum(cal_dict[key]["accuracy"]))
        confidence_scores.append(int(sum(cal_dict[key]["confidence"])))

        if max_examples_per_column < total_preds[-1]:
            max_examples_per_column = total_preds[-1]

    x = np.arange(len(labels))  # the label locations
    width = 0.3

    # x = np.arange(0, len(labels) * 4, 4)  # the label locations + increase space between bars
    # width = 1  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, total_preds, width, label='All examples per bin')
    rects2 = ax.bar(x, correct_preds, width, label='# correct prediction per bin')
    rects3 = ax.bar(x + width, confidence_scores, width, label='Total confidence scores')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Confidence score')
    ax.set_ylabel('Number of examples')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, max_examples_per_column + 100])
    ax.grid(True)
    ax.legend(prop={'size': 9})

    # Increase font size for this figure
    ax.title.set_fontsize(12)
    for item in ([ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(10)

    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(9)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # fig.tight_layout()
    fig.set_size_inches(11.2, 4.8)

    plt.savefig(file_name)

def plot_calibration_figure_type2(cal_dict, calibration_error, title, file_name):
    labels = []
    accuracy_avg_scores = []
    confidence_avg_scores = []

    for key in sorted(cal_dict.keys()):
        labels.append(key)

        if len(cal_dict[key]["accuracy"]) > 0:
            accuracy_avg_scores.append(sum(cal_dict[key]["accuracy"]) / len(cal_dict[key]["accuracy"]))
            confidence_avg_scores.append(sum(cal_dict[key]["confidence"]) / len(cal_dict[key]["confidence"]))
        else:
            accuracy_avg_scores.append(0)
            confidence_avg_scores.append(0)

    fig, axs = plt.subplots()
    # fig.suptitle(title)
    axs.set_title(title)

    # print(accuracy_avg_scores)
    # print(confidence_avg_scores)

    x = np.linspace(0, 1.0, len(cal_dict.keys()) + 1)

    axs.bar(x[:-1], accuracy_avg_scores, width=np.diff(x), align="edge", ec="k", color="#ff8886")
    axs.plot(np.arange(0, 2, 1), '--')
    plt.text(0, 0.9, "ECE: " + str(round(calibration_error * 100, 2)) + "%", size=10, rotation=0.,
             ha="center", va="center",
             bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))

    # for i in range(20):
    #     plt.text(arr[1][i], arr[0][i], str(int(arr[0][i])))

    # axs.set_xscale("log")
    axs.set_xlim([0, 1.0])
    axs.set_ylim([0, 1.0])
    # axs.grid(True)

    axs.set(xlabel='Confidence', ylabel='Accuracy')

    plt.savefig(file_name)

    # fig.tight_layout()
    fig.set_size_inches(4.8, 4.8)

    plt.savefig(file_name)


if __name__ == '__main__':

    # get_results()
    get_multiple_runs_results()

    # times = ["1st", "2nd", "3rd", "4th", "5th"]
    # for time in times:
    #     compute_calibration_error(bucket_size=100, task_name="RTE", time=time)

    # times = ["1st", "2nd", "3rd", "4th", "5th"]
    # bucket_sizes = [5, 10, 15, 20, 25, 30]
    # task_names = ["RTE", "QQP", "MRPC", "SST-2", "QNLI"]
    #
    # for bucket_size in bucket_sizes:
    #     for task_name in task_names:
    #         performances = []
    #
    #         for time in times:
    #             baseline, extra = compute_calibration_error(bucket_size=bucket_size, task_name=task_name, time=time)
    #             if len(performances) == 0:
    #                 performances.extend([str(baseline), str(extra)])
    #             else:
    #                 performances.append(str(extra))
    #
    #         print(str(bucket_size) + " --- " + str(task_name))
    #         print("\t".join(performances))



