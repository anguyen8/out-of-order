import pickle
import numpy as np
import collections

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from scipy import spatial
from texttable import Texttable
import nlp
from tqdm import tqdm

from os import makedirs, mkdir, listdir
from os.path import isfile, join, exists
import random
import statistics
from shuffler import *

base_dir = "../../examples/models/"

label_dict = {"CoLA": []}
csfont = {'fontname':'Times New Roman'}


def draw_one_figure(cos_sim_pred, file_name, task_name, for_paper=False):
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    # axs.autoscale()
    fig.set_size_inches(9.6, 6.4)

    # Title will be updated later in paper.
    if not for_paper:
        fig.suptitle("GLUE - " + task_name)

    arr = axs.hist(cos_sim_pred, bins=np.linspace(-1.0, 1.0, 21), histtype='bar', ec='black')

    # No need to show this information in paper.
    if not for_paper:
        for i in range(20):
            plt.text(arr[1][i], arr[0][i], str(int(arr[0][i])))

    axs.set_xlim([-1.0, 1.0])
    # axs.set_ylim([0, 25000])
    axs.grid(True)

    if not for_paper:
        # axs.set(xlabel='Cosine Similarity', ylabel='Occurrence (Sentence Level)')
        plt.xlabel("Cosine Similarity", fontsize=15)
        plt.ylabel("Occurrence (Sentence Level)", fontsize=15)
    else:
        # axs.set(xlabel='Cosine Similarity', ylabel='Occurrence')
        plt.xlabel("Cosine Similarity", fontsize=15)
        plt.ylabel("Occurrence", fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

    plt.savefig(file_name)
    # plt.show()

def draw_two_figures(cos_sim_same_pred, cos_sime_diff_pred, file_name, task_name, for_paper=False):
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    # Title will be updated later in paper.
    if not for_paper:
        fig.suptitle("GLUE - " + task_name)

    axs = axs.ravel()

    # Note: bins = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    arr0 = axs[0].hist(cos_sim_same_pred, bins=np.linspace(-1.0, 1.0, 21), histtype='bar', ec='black')
    # No need to show this information in paper.
    if not for_paper:
        for i in range(20):
            axs[0].text(arr0[1][i], arr0[0][i], str(int(arr0[0][i]))).set_fontsize(6)

    arr1 = axs[1].hist(cos_sime_diff_pred, bins=np.linspace(-1.0, 1.0, 21), histtype='bar', ec='black')
    # No need to show this information in paper.
    if not for_paper:
        for i in range(20):
            axs[1].text(arr1[1][i], arr1[0][i], str(int(arr1[0][i]))).set_fontsize(6)

    # Will be updated later in paper.
    if not for_paper:
        axs[0].set_title('Prediction SAME')
        axs[1].set_title('Prediction CHANGE')

    axs[0].set_xlim([-1.0, 1.0])
    # axs[0].set_ylim([0, 25000])
    axs[0].grid(True)

    axs[1].set_xlim([-1.0, 1.0])
    # axs[1].set_ylim([0, 25000])
    axs[1].grid(True)

    for ax in axs.flat:
        if not for_paper:
            # ax.set(xlabel='Cosine Similarity', ylabel='Occurrence (Token Level)')
            ax.set(xlabel='Cosine Similarity', ylabel='Occurrence (Sentence Level)')
        else:
            ax.set(xlabel='Cosine Similarity', ylabel='Occurrence')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(file_name)
def draw_sentiment_histogram(labels, grounth_truths, preds, correct, file_name, title):
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, grounth_truths, width, label='GT')
    rects2 = ax.bar(x, preds, width, label='Prediction')
    rects3 = ax.bar(x + width, correct, width, label='Correct')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Sentiments')
    ax.set_ylabel('Occurrence (Sentence Level)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1100])
    ax.grid(True)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    plt.savefig(file_name)

def draw_confidence_histogram(labels, conf_correct, conf_correct_shuffled, file_name, title):
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, conf_correct, width, label='Original data')
    rects2 = ax.bar(x + width / 2, conf_correct_shuffled, width, label='Shuffled data')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Sentiments')
    ax.set_ylabel('Average Confidence Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1.0])
    ax.grid(True)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.savefig(file_name)
def draw_confidence_histogram_one_plot(labels, conf_correct, file_name, title):
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, conf_correct, width, label='SSE v2 data')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Sentiments')
    ax.set_ylabel('Average Confidence Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1.0])
    ax.grid(True)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)

    fig.tight_layout()

    plt.savefig(file_name)

def compute_calibration_error(bucket_size):
    epoch = 1300
    is_token_cls = False
    write_files = True
    index = 0

    # SE v1
    # ml_objects_path = "../datgru/model_files/bert-san-rest_total-finetune-sentence_att_original/2020-05-29_16-01-57_original_se_v1/" # Previously w/o confidence score: 2020-05-10_13-03-00
    # ml_objects_shuffled_path = "../datgru/model_files/bert-san-rest_total-finetune-sentence_att_original/2020-05-29_16-05-37_shuffled_se_v1/" # Previously w/o confidence score: 2020-05-10_13-28-34

    # SSE v1
    ml_objects_path = "../datgru/model_files/bert-san-rest_total-finetune-sentence_att_original/2020-05-26_10-08-57_eval_best_added_confidence_score/"
    ml_objects_shuffled_path = "../datgru/model_files/bert-san-rest_total-finetune-sentence_att_original/2020-05-26_10-09-36_eval_best_added_confidence_score/"

    # SSE v2
    # ml_objects_path = "../datgru/model_files/bert-san-rest_total-finetune-sentence_att_original/2020-05-27_09-12-04_original_semeval_bert_v2/"
    # ml_objects_shuffled_path = "../datgru/model_files/bert-san-rest_total-finetune-sentence_att_original/2020-05-27_09-12-39_shuffled_semeval_bert_v2/"

    # SSE v3
    # ml_objects_path = "../datgru/model_files/bert-san-rest_total-finetune-sentence_att_original/2020-05-27_09-13-55_original_semeval_bert_v3/"
    # ml_objects_shuffled_path = "../datgru/model_files/bert-san-rest_total-finetune-sentence_att_original/2020-05-27_09-14-33_shuffled_semeval_bert_v3/"

    with open(ml_objects_path + "ml_objects_test_" + str(epoch) + ".pickle", "rb") as file_path:
        ml_objects = pickle.load(file_path)

    with open(ml_objects_shuffled_path + "ml_objects_test_shuffled_" + str(epoch) + ".pickle", "rb") as file_path:
        ml_objects_shuffled = pickle.load(file_path)

    calibration_error_original, calibration_error_shuffled = 0, 0
    confidence_range = 1 / bucket_size
    cal_dict_original, cal_dict_shuffled = {}, {}

    for i in range(bucket_size):
        cal_dict_original[round((i + 1) * confidence_range, 2)] = {"accuracy": [], "confidence": []}
        cal_dict_shuffled[round((i + 1) * confidence_range, 2)] = {"accuracy": [], "confidence": []}

    for obj, obj_shuffled in zip(ml_objects, ml_objects_shuffled):

        confidence_score_orginal = np.max(obj.get_confidence_score())
        confidence_score_shuffled = np.max(obj_shuffled.get_confidence_score())

        flag_original, flag_shuffled = True, True

        for key in sorted(cal_dict_original.keys()):

            if flag_original and confidence_score_orginal <= key:
                cal_dict_original[key]["confidence"].append(confidence_score_orginal)

                if obj.get_pred_label() == obj.get_ground_truth():
                    cal_dict_original[key]["accuracy"].append(1)
                else:
                    cal_dict_original[key]["accuracy"].append(0)

                flag_original = False

            if flag_shuffled and confidence_score_shuffled <= key:
                cal_dict_shuffled[key]["confidence"].append(confidence_score_shuffled)

                if obj_shuffled.get_pred_label() == obj.get_ground_truth():
                    cal_dict_shuffled[key]["accuracy"].append(1)
                else:
                    cal_dict_shuffled[key]["accuracy"].append(0)

                flag_shuffled = False

            if not flag_original and not flag_shuffled:
                break

    accuracy_avg_list_original, confidence_avg_list_original = [], []
    accuracy_avg_list_shuffled, confidence_avg_list_shuffled = [], []

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
        cal_per_bin_shuffled, accuracy_avg_shuffled, confidence_avg_shuffled = compute_cal_error_per_bin(
            bucket_ith=cal_dict_shuffled[key], n_samples=len(ml_objects))

        # Sum all calibration errors from all buckets
        calibration_error_original += cal_per_bin_original
        calibration_error_shuffled += cal_per_bin_shuffled

        accuracy_avg_list_original.append(accuracy_avg_original)
        confidence_avg_list_original.append(confidence_avg_original)

        accuracy_avg_list_shuffled.append(accuracy_avg_shuffled)
        confidence_avg_list_shuffled.append(confidence_avg_shuffled)

    print("Calibration Error (Original): " + str(round(calibration_error_original * 100, 2)))
    print("Calibration Error (Shuffled): " + str(round(calibration_error_shuffled * 100, 2)))

    # plot_calibration_figure_type1(cal_dict_original, title="Expected Calibration Error (SE)", file_name="calibration_error_se.jpg")
    plot_calibration_figure_type1(cal_dict_shuffled, title="Expected Calibration Error (SSE v1) Shuffled",
                                  file_name="calibration_error_sse_v1_shuffled.jpg")

    # plot_calibration_figure_type2(cal_dict_original, calibration_error_original, title="Expected Calibration Error (SE)", file_name="calibration_error_type2_se.jpg")
    plot_calibration_figure_type2(cal_dict_shuffled, calibration_error_shuffled,
                                  title="Expected Calibration Error (SSE v1) Shuffled",
                                  file_name="calibration_error_type2_sse_v1_shuffled.jpg")
def plot_calibration_figure_type1(cal_dict, title, file_name):
    labels = []
    total_preds = []
    correct_preds = []
    confidence_scores = []

    for key in sorted(cal_dict.keys()):
        labels.append(key)
        total_preds.append(len(cal_dict[key]["accuracy"]))
        correct_preds.append(sum(cal_dict[key]["accuracy"]))
        confidence_scores.append(int(sum(cal_dict[key]["confidence"])))

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
    ax.set_ylim([0, 900])
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
    # fig.suptitle("SemEval2014 Calibration results")
    axs.set_title(title)

    print(accuracy_avg_scores)
    print(confidence_avg_scores)

    x = np.linspace(0, 1.0, len(cal_dict.keys()) + 1)

    axs.bar(x[:-1], accuracy_avg_scores, width=np.diff(x), align="edge", ec="k", color="#ff8886")
    axs.plot(np.arange(0, 2, 1), '--')
    plt.text(0, 0.9, "ECE: " + str(round(calibration_error * 100, 2)) + "%", size=10, rotation=0.,
             ha="center", va="center",
             bbox=dict(boxstyle="round",
                       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),
                       ))

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


def load_and_print_out_predictions(task, labels):
    path = base_dir + "roberta-base/" + task + "/evaluation/original/ml_objects.pickle"
    path_unigram = base_dir + "roberta-base/" + task + "/evaluation/shuffled/ml_objects.pickle"
    path_bigram = base_dir + "roberta-base/" + task + "/evaluation/shuffled_bigram/ml_objects.pickle"
    ml_trigram = base_dir + "roberta-base/" + task + "/evaluation/shuffled_trigram/ml_objects.pickle"

    objs_original, objs_unigram, objs_bigram, objs_trigram = [], [], [], []

    with open(path, "rb") as file_path:
        objs_original = pickle.load(file_path)
    with open(path_unigram, "rb") as file_path:
        objs_unigram = pickle.load(file_path)
    with open(path_bigram, "rb") as file_path:
        objs_bigram = pickle.load(file_path)
    with open(ml_trigram, "rb") as file_path:
        objs_trigram = pickle.load(file_path)

    if len(objs_original) <= 0:
        return

    pred, pred_correct, pred_wrong = {}, {}, {}
    conf_correct = {}

    for obj in objs_original:

        # ------------------------------------------------------------------
        if obj.get_pred_label() not in pred:
            pred[obj.get_pred_label()] = 0
        pred[obj.get_pred_label()] += 1

        if obj.get_pred_label() == obj.get_ground_truth():
            print("Sentence: " + str(obj.get_tokens()))
            print("Prediction: " + str(labels[int(obj.get_pred_label())]))
            print("Ground truth: " + str(obj.get_ground_truth()) + "\n")

            if obj.get_pred_label() not in pred_correct:
                pred_correct[obj.get_pred_label()] = 0
                conf_correct[obj.get_pred_label()] = 0

            pred_correct[obj.get_pred_label()] += 1
            conf_correct[obj.get_pred_label()] += np.max(obj.get_confidence_score())
        else:
            if obj.get_pred_label() not in pred_wrong:
                pred_wrong[obj.get_pred_label()] = 0

            pred_wrong[obj.get_pred_label()] += 1
        # ------------------------------------------------------------------

    # labels = ["positive", "negative", "neutral", "conflict"]
    # grounth_truths = [657, 222, 94, 52]
    # preds_ori, correct_ori = [], []
    # conf_correct_ori = []
    #
    # for i in labels:
    #     # Original data
    #     preds_ori.append(pred[i] if i in pred else 0)
    #     correct_ori.append(pred_correct[i] if i in pred_correct else 0)
    #     conf_correct_ori.append(round(conf_correct[i] / pred_correct[i], 2) if i in conf_correct else 0)
    #
    # draw_sentiment_histogram(labels, grounth_truths, preds_ori, correct_ori,
    #                          file_name="test_plot_semeval_predictions_sse_v2_on_sse_v2.jpg",
    #                          title="SemEval2014 SSE v2 dataset - Artificial BERT fine-tuned on SemEval data")
    #
    # draw_confidence_histogram_one_plot(labels, conf_correct_ori,
    #                                    file_name="test_plot_semeval_confidence_sse_v2_on_sse_v2.jpg",
    #                                    title="SemEval2014 SSE v2 dataset - Artificial BERT fine-tuned on SemEval data")

    print("All predictions" + str(pred))
    print("Correct predictions" + str(pred_correct))
    print("Wrong predictions" + str(pred_wrong))


def load_and_visualize_representations(task_name, labels, write_files=False):

    data_dir = base_dir + "roberta-base/" + task_name + "/evaluation/"
    path = data_dir + "original/" + "ml_objects.pickle"
    path_unigram = data_dir + "shuffled/" + "ml_objects.pickle"
    path_bigram = data_dir + "shuffled_bigram/" + "ml_objects.pickle"
    path_trigram = data_dir + "shuffled_trigram/" + "ml_objects.pickle"

    with open(path, "rb") as file_path:
        objs_original = pickle.load(file_path)
    with open(path_unigram, "rb") as file_path:
        objs_unigram = pickle.load(file_path)
    with open(path_bigram, "rb") as file_path:
        objs_bigram = pickle.load(file_path)
    with open(path_trigram, "rb") as file_path:
        objs_trigram = pickle.load(file_path)

    if len(objs_original) <= 0:
        return

    index = 0
    if write_files:
        file_path = open(ml_objects_path + "output_double_check_sse_only.txt", "w")
    else:
        file_path = open(ml_objects_path + "output_temp.txt", "w")

    cos_sim_pred, cos_sim_same_pred, cos_sim_diff_pred = [], [], []

    pred_ori, pred_shuffled = {}, {}
    pred_ori_correct, pred_shuffled_correct = {}, {}
    conf_ori_correct, conf_shuffled_correct = {}, {}

    for obj, obj_shuffled in zip(ml_objects, ml_objects_shuffled):

        if write_files:
            file_path.write("Sample " + str(index) + "\n")

        cos_sim = 1 - spatial.distance.cosine(obj.get_sentence_representation(), obj_shuffled.get_sentence_representation())
        cos_sim_pred.append(cos_sim)

        if write_files:
            file_path.write("Sentence: " + " ".join(obj.get_tokens()) + "\n")
            file_path.write("Sentence shuffled: " + " ".join(obj_shuffled.get_tokens()) + "\n")
            file_path.write("Prediction: " + str(obj.get_pred_label()) + "\n")
            file_path.write("Prediction shuffled: " + str(obj_shuffled.get_pred_label()) + "\n")
            file_path.write("Ground truth: " + str(obj.get_ground_truth()) + "\n")
            file_path.write("Cosine Similarity Score between 2 sentence representations: " + str(cos_sim) + "\n")

        # if obj.get_pred_label() != obj_shuffled.get_pred_label() and cos_sim >= 0.9:  # same_embed_diff_pred:
        if obj.get_pred_label() == obj_shuffled.get_pred_label():
            cos_sim_same_pred.append(cos_sim)
            print_table = True
        else:
            cos_sim_diff_pred.append(cos_sim)
            print_table = False

        # ------------------------------------------------------------------
        if obj.get_pred_label() not in pred_ori:
            pred_ori[obj.get_pred_label()] = 0
        pred_ori[obj.get_pred_label()] += 1

        if obj_shuffled.get_pred_label() not in pred_shuffled:
            pred_shuffled[obj_shuffled.get_pred_label()] = 0
        pred_shuffled[obj_shuffled.get_pred_label()] += 1

        if obj.get_pred_label() == obj.get_ground_truth():
            if obj.get_pred_label() not in pred_ori_correct:
                pred_ori_correct[obj.get_pred_label()] = 0
                conf_ori_correct[obj.get_pred_label()] = 0

            pred_ori_correct[obj.get_pred_label()] += 1
            conf_ori_correct[obj.get_pred_label()] += np.max(obj.get_confidence_score())

        if obj_shuffled.get_pred_label() == obj.get_ground_truth():
            if obj_shuffled.get_pred_label() not in pred_shuffled_correct:
                pred_shuffled_correct[obj_shuffled.get_pred_label()] = 0
                conf_shuffled_correct[obj_shuffled.get_pred_label()] = 0

            pred_shuffled_correct[obj_shuffled.get_pred_label()] += 1
            conf_shuffled_correct[obj_shuffled.get_pred_label()] += np.max(obj_shuffled.get_confidence_score())
        # ------------------------------------------------------------------

        if print_table:
            print("Sample " + str(index))
            print("Sentence: " + " ".join(obj.get_tokens()))
            print("Sentence shuffled: " + " ".join(obj_shuffled.get_tokens()))
            print("Prediction: " + str(obj.get_pred_label()))
            print("Prediction shuffled: " + str(obj_shuffled.get_pred_label()))
            print("Ground truth: " + str(obj.get_ground_truth()))
            print("Cosine Similarity Score between 2 sentence representations: " + str(cos_sim) + "\n")

        if write_files:
            file_path.write("\n")

        index += 1

    if write_files:
        draw_one_figure(cos_sim_pred, file_name="test_plot_semeval_merged_se_v1.jpg")
        draw_two_figures(cos_sim_same_pred, cos_sim_diff_pred, file_name="test_plot_semeval_se_v1.jpg")

    labels = ["positive", "negative", "neutral", "conflict"]
    grounth_truths = [657, 222, 94, 52]
    preds_ori, correct_ori = [], []
    preds_shuffled, correct_shuffled = [], []
    conf_correct_ori, conf_correct_shuffled = [], []

    for i in labels:
        # Original data
        preds_ori.append(pred_ori[i] if i in pred_ori else 0)
        correct_ori.append(pred_ori_correct[i] if i in pred_ori_correct else 0)

        # Shuffled data
        preds_shuffled.append(pred_shuffled[i] if i in pred_shuffled else 0)
        correct_shuffled.append(pred_shuffled_correct[i] if i in pred_shuffled_correct else 0)

        conf_correct_ori.append(round(conf_ori_correct[i] / pred_ori_correct[i], 2) if i in conf_ori_correct else 0)
        conf_correct_shuffled.append(
            round(conf_shuffled_correct[i] / pred_shuffled_correct[i], 2) if i in conf_shuffled_correct else 0)

    draw_sentiment_histogram(labels, grounth_truths, preds_ori, correct_ori,
                             file_name="test_plot_semeval_predictions_original_se_v1.jpg",
                             title="SemEval2014 Original dataset - Original BERT fine-tuned on SemEval data")

    draw_sentiment_histogram(labels, grounth_truths, preds_shuffled, correct_shuffled,
                             file_name="test_plot_semeval_predictions_shuffled_se_v1.jpg",
                             title="SemEval2014 Shuffled dataset - Original BERT fine-tuned on SemEval data")

    draw_confidence_histogram(labels, conf_correct_ori, conf_correct_shuffled,
                              file_name="test_plot_semeval_confidence_se_v1.jpg",
                              title="SemEval2014 dataset - Original BERT fine-tuned on SemEval data")

    print("All predictions (Original)" + str(pred_ori))
    print("Correct predictions (Original)" + str(pred_ori_correct))
    print("All predictions (Shuffled)" + str(pred_shuffled))
    print("Correct predictions (Shuffled)" + str(pred_shuffled_correct))

avg_ori_conf_scores, avg_uni_shuffled_conf_scores, avg_bi_shuffled_conf_scores, avg_tri_shuffled_conf_scores, avg_swapped_conf_scores = [], [], [], [], []
def analyze_results_for_confidence_scores(task_name, labels, base_model, seed="", sub_folder=None, write_files=False, for_paper=False, conf_mode=False):

    # FOR Random seeds = [100, 200, 300, ..., 1000]
    # data_dir = base_dir + base_model + "/" + task_name + "/evaluation/" + (seed + "/" if seed else "") + (sub_folder + "/" if sub_folder is not None else "")
    # FOR Random seeds = ["1st_42", "2nd_100", "3rd_200", "4th_300", "5th_400"]
    # data_dir = base_dir + base_model + "/" + seed + "/" + task_name + "/evaluation/" + (sub_folder + "/" if sub_folder is not None else "")

    data_dir = base_dir + base_model + "/" + task_name + "/evaluation/"

    output_dir = base_dir + base_model + "/" + "analysis_422/"

    if not exists(output_dir):
        makedirs(output_dir)

    path =          data_dir + "original/"          + "ml_objects.pickle"
    path_unigram =  data_dir + "shuffled/"          + "ml_objects.pickle"
    path_bigram =   data_dir + "shuffled_bigram/"   + "ml_objects.pickle"
    path_trigram =  data_dir + "shuffled_trigram/"  + "ml_objects.pickle"
    path_swapped =  data_dir + "422/swapped_nouns/"     + "ml_objects.pickle"

    # with open(path, "rb") as file_path:
    #     objs_original = pickle.load(file_path)
    # with open(path_unigram, "rb") as file_path:
    #     objs_unigram = pickle.load(file_path)
    # with open(path_bigram, "rb") as file_path:
    #     objs_bigram = pickle.load(file_path)
    # with open(path_trigram, "rb") as file_path:
    #     objs_trigram = pickle.load(file_path)
    with open(path_swapped, "rb") as file_path:
        objs_swapped = pickle.load(file_path)

    # if len(objs_original) <= 0:
    #     return

    if not conf_mode:
        cos_sim_pred = {"unigram": [], "bigram": [], "trigram": [], "swapped": []}
        cos_sim_same_pred = {"unigram": [], "bigram": [], "trigram": [], "swapped": []}
        cos_sim_diff_pred = {"unigram": [], "bigram": [], "trigram": [], "swapped": []}

        if write_files:
            file_path = open(output_dir + "output_analysis_" + task_name + ".txt", "w")

        # Sort the list of objects by the following criteria
        # + Highest cosine similarity score (original vs unigram)
        # + Same prediction (original vs unigram)
        for obj, obj_uni, obj_bi, obj_tri, obj_swapped in zip(objs_original, objs_unigram, objs_bigram, objs_trigram, objs_swapped):
            cos_sim_uni = 1 - spatial.distance.cosine(obj.get_sentence_representation(), obj_uni.get_sentence_representation())
            cos_sim_bi  = 1 - spatial.distance.cosine(obj.get_sentence_representation(), obj_bi.get_sentence_representation())
            cos_sim_tri = 1 - spatial.distance.cosine(obj.get_sentence_representation(), obj_tri.get_sentence_representation())
            cos_sim_swap = 1 - spatial.distance.cosine(obj.get_sentence_representation(), obj_swapped.get_sentence_representation())

            cos_sim_uni = round(cos_sim_uni, 2)
            cos_sim_bi = round(cos_sim_bi, 2)
            cos_sim_tri = round(cos_sim_tri, 2)
            cos_sim_swap = round(cos_sim_swap, 2)

            obj_uni.set_cos_sim(cos_sim_uni)
            obj_bi.set_cos_sim(cos_sim_bi)
            obj_tri.set_cos_sim(cos_sim_tri)
            obj_swapped.set_cos_sim(cos_sim_swap)

            cos_sim_pred["unigram"].append(cos_sim_uni)
            if obj.get_pred_label() == obj_uni.get_pred_label(): cos_sim_same_pred["unigram"].append(cos_sim_uni)
            else: cos_sim_diff_pred["unigram"].append(cos_sim_uni)

            cos_sim_pred["bigram"].append(cos_sim_bi)
            if obj.get_pred_label() == obj_bi.get_pred_label(): cos_sim_same_pred["bigram"].append(cos_sim_bi)
            else: cos_sim_diff_pred["bigram"].append(cos_sim_bi)

            cos_sim_pred["trigram"].append(cos_sim_tri)
            if obj.get_pred_label() == obj_tri.get_pred_label(): cos_sim_same_pred["trigram"].append(cos_sim_tri)
            else: cos_sim_diff_pred["trigram"].append(cos_sim_tri)

            cos_sim_pred["swapped"].append(cos_sim_swap)
            if obj.get_pred_label() == obj_swapped.get_pred_label(): cos_sim_same_pred["swapped"].append(cos_sim_swap)
            else: cos_sim_diff_pred["swapped"].append(cos_sim_swap)

        # Compute the sorted indices
        sorted_indices = []
        # for i in range(len(objs_unigram)):
        #     sorted_indices.append([objs_unigram[i], i])
        # sorted_indices.sort(reverse=True)

        # for i in range(len(objs_swapped)):
        #     sorted_indices.append([objs_swapped[i], i])
        # sorted_indices.sort()
        # sorted_indices = [element[1] for element in sorted_indices]

        sorted_indices = range(len(objs_unigram))

        # Write all examples in cosine-score-sorted order by descending
        if write_files:
            for idx in sorted_indices:

                skipped = False
                # if task_name == "RTE" or task_name == "QNLI":
                #     if objs_original[idx].get_ground_truth() == objs_swapped[idx].get_pred_label() == 0: # positive labels
                #         skipped = False
                # else:
                #     if objs_original[idx].get_ground_truth() == objs_swapped[idx].get_pred_label() == 1: # positive labels
                #         skipped = False

                if objs_original[idx].get_tokens() == objs_swapped[idx].get_tokens():
                    skipped = True

                if skipped:
                    continue

                file_path.write("*********** Sample " + str(idx) + " ***********\n")

                file_path.write("Sentence: " + str(objs_original[idx].get_sentences()) + "\n")
                file_path.write("Shuffled sentences: " + "\n")
                # file_path.write("\t+ Unigram: " + str(objs_unigram[idx].get_sentences()) + "\n")
                # file_path.write("\t+ Bigram: " + str(objs_bigram[idx].get_sentences()) + "\n")
                # file_path.write("\t+ Trigram: " + str(objs_trigram[idx].get_sentences()) + "\n")
                file_path.write("\t+ Swapped: " + str(objs_swapped[idx].get_sentences()) + "\n")

                if labels:
                    file_path.write("Prediction: " + str(labels[int(objs_original[idx].get_pred_label())]) + "(" + str(objs_original[idx].get_pred_label()) + ")" + "\n")
                    file_path.write("Predictions with shuffle: " + "\n")
                    # file_path.write("\t+ Unigram: " + str(labels[int(objs_unigram[idx].get_pred_label())]) + "(" + str(objs_unigram[idx].get_pred_label()) + ")" + "\n")
                    # file_path.write("\t+ Bigram: " + str(labels[int(objs_bigram[idx].get_pred_label())]) + "(" + str(objs_bigram[idx].get_pred_label()) + ")" + "\n")
                    # file_path.write("\t+ Trigram: " + str(labels[int(objs_trigram[idx].get_pred_label())]) + "(" + str(objs_trigram[idx].get_pred_label()) + ")" + "\n")
                    file_path.write("\t+ Swapped: " + str(labels[int(objs_swapped[idx].get_pred_label())]) + "(" + str(objs_swapped[idx].get_pred_label()) + ")" + "\n")

                    file_path.write("Ground truth: " + str(labels[int(objs_original[idx].get_ground_truth())]) + "\n")
                else:
                    file_path.write("Prediction: " + str(objs_original[idx].get_pred_label()) + "\n")
                    file_path.write("Predictions with shuffle: " + "\n")
                    # file_path.write("\t+ Unigram: " + str(objs_unigram[idx].get_pred_label()) + "\n")
                    # file_path.write("\t+ Bigram: " + str(objs_bigram[idx].get_pred_label()) + "\n")
                    # file_path.write("\t+ Trigram: " + str(objs_trigram[idx].get_pred_label()) + "\n")
                    file_path.write("\t+ Swapped: " + str(objs_swapped[idx].get_pred_label()) + "\n")

                    file_path.write("Ground truth: " + str(objs_original[idx].get_ground_truth()) + "\n")

                file_path.write("Cosine Similarity: " + "\n")
                # file_path.write("\t+ Orginal vs Unigram: " + str(objs_unigram[idx].get_cos_sim()) + "\n")
                # file_path.write("\t+ Orginal vs Bigram: " + str(objs_bigram[idx].get_cos_sim()) + "\n")
                # file_path.write("\t+ Orginal vs Trigram: " + str(objs_trigram[idx].get_cos_sim()) + "\n")
                file_path.write("\t+ Orginal vs Swapped: " + str(objs_swapped[idx].get_cos_sim()) + "\n")

                file_path.write("\n")

            file_path.close()

        modes = ["unigram", "bigram", "trigram"]
        for mode in modes:
            # ext = ".jpg"
            ext = ".jpg" if not for_paper else ".pdf"
            # draw_one_figure(cos_sim_pred[mode], file_name=output_dir + "plot_merged_" + task_name + "_" + mode + ext, task_name=task_name, for_paper=for_paper)
            # draw_two_figures(cos_sim_same_pred[mode], cos_sim_diff_pred[mode], file_name=output_dir + "plot_" + task_name + "_" + mode + ext, task_name=task_name, for_paper=for_paper)
    else:
        global avg_ori_conf_scores, avg_uni_shuffled_conf_scores, avg_bi_shuffled_conf_scores, avg_tri_shuffled_conf_scores, avg_swapped_conf_scores

        # HOT FIX to print out all confidence scores 09/08/2020
        ori_conf_scores, uni_shuffled_conf_scores, bi_shuffled_conf_scores, tri_shuffled_conf_scores, swapped_conf_scores = [], [], [], [], []

        # for obj, obj_uni, obj_bi, obj_tri, obj_swapped in zip(objs_original, objs_unigram, objs_bigram, objs_trigram, objs_swapped):
        # for obj, obj_uni in zip(objs_original, objs_unigram):
        for obj_swapped in objs_swapped:
            # ThangPM: Only consider examples that were incorrectly predicted. (01-25-2021)
            # if obj.get_pred_label() != obj.get_ground_truth():
            # ori_conf_scores.append(np.max(obj.get_confidence_score()))

            # ThangPM: Only consider examples that were incorrectly predicted. (01-25-2021)
            # if obj_uni.get_pred_label() != obj_uni.get_ground_truth():
            #     uni_shuffled_conf_scores.append(np.max(obj_uni.get_confidence_score()))

            # For MiniQ1 Table
            # ori_conf_scores.append(np.max(obj.get_confidence_score()))
            # uni_shuffled_conf_scores.append(np.max(obj_uni.get_confidence_score()))
            # bi_shuffled_conf_scores.append(np.max(obj_bi.get_confidence_score()))
            # tri_shuffled_conf_scores.append(np.max(obj_tri.get_confidence_score()))
            swapped_conf_scores.append(np.max(obj_swapped.get_confidence_score()))

        # avg_ori_conf_scores.append(round(statistics.mean(ori_conf_scores) * 100, 2))
        # avg_uni_shuffled_conf_scores.append(round(statistics.mean(uni_shuffled_conf_scores) * 100, 2))
        # avg_bi_shuffled_conf_scores.append(round(statistics.mean(bi_shuffled_conf_scores) * 100, 2))
        # avg_tri_shuffled_conf_scores.append(round(statistics.mean(tri_shuffled_conf_scores) * 100, 2))
        avg_swapped_conf_scores.append(round(statistics.mean(swapped_conf_scores) * 100, 2))

        # print("len(original_incorrect_examples) = " + str(len(ori_conf_scores)))
        # print("len(unigram_incorrect_examples) = " + str(len(uni_shuffled_conf_scores)))

        if seed == "1000" or seed == "5th_400" or seed == "422":
            # print("Original")
            # print("\t".join(list([str(item) for item in avg_ori_conf_scores])))
            # print("Unigram")
            # print("\t".join(list([str(item) for item in avg_uni_shuffled_conf_scores])))
            # print("Bigram")
            # print("\t".join(list([str(item) for item in avg_bi_shuffled_conf_scores])))
            # print("Trigram")
            # print("\t".join(list([str(item) for item in avg_tri_shuffled_conf_scores])))
            print("Swapped")
            print("\t".join(list([str(item) for item in avg_swapped_conf_scores])))

            # avg_ori_conf_scores.clear()
            # avg_uni_shuffled_conf_scores.clear()
            # avg_bi_shuffled_conf_scores.clear()
            # avg_tri_shuffled_conf_scores.clear()
            avg_swapped_conf_scores.clear()

def analyze_statistics_for_only_stsb(task_name, base_model):

    data_dir = base_dir + base_model + "/" + task_name + "/evaluation_2nd/"

    path =          data_dir + "original/"          + "ml_objects.pickle"
    path_unigram =  data_dir + "shuffled/"          + "ml_objects.pickle"
    path_bigram =   data_dir + "shuffled_bigram/"   + "ml_objects.pickle"
    path_trigram =  data_dir + "shuffled_trigram/"  + "ml_objects.pickle"

    with open(path, "rb") as file_path:
        objs_original = pickle.load(file_path)
    with open(path_unigram, "rb") as file_path:
        objs_unigram = pickle.load(file_path)
    with open(path_bigram, "rb") as file_path:
        objs_bigram = pickle.load(file_path)
    with open(path_trigram, "rb") as file_path:
        objs_trigram = pickle.load(file_path)

    if len(objs_original) <= 0:
        return

    stats = {"ground_truth": {}, "original": {}, "unigram": {}, "bigram": {}, "trigram": {}}
    for key in stats.keys():
        for i in range(6):
            stats[key][i] = 0

    for obj, obj_uni, obj_bi, obj_tri in zip(objs_original, objs_unigram, objs_bigram, objs_trigram):
        stats["ground_truth"][round(obj.get_ground_truth())] += 1
        stats["original"][round(obj.get_pred_label())] += 1
        stats["unigram"][round(obj_uni.get_pred_label())] += 1
        stats["bigram"][round(obj_bi.get_pred_label())] += 1
        stats["trigram"][round(obj_tri.get_pred_label())] += 1

    [print(str(key) + ":" + str(values)) for key, values in stats.items()]

def load_stored_objects(file_path):
    with open(file_path, "rb") as input_file:
        list_objects = pickle.load(input_file)

    return list_objects
def analyze_results_for_calibration(task_list, label_dict, mode):

    tasks_dict = {}
    score_dict = {}
    prediction_stats = {}

    for task_name in task_list:
        data_dir = base_dir + "roberta-base/" + task_name + "/evaluation/"
        original_path = data_dir + "original" + "/" + "ml_objects.pickle"
        shuffled_path = data_dir + mode + "/" + "ml_objects.pickle"

        list_original_objs = load_stored_objects(original_path)
        list_shuffled_objs = load_stored_objects(shuffled_path)

        tasks_dict[task_name] = [list_original_objs, list_shuffled_objs]
        score_dict[task_name] = {"score": 0, "total": len(list_original_objs)}

        for label in label_dict[task_name]:
            if task_name not in prediction_stats:
                prediction_stats[task_name] = {}

            prediction_stats[task_name][label] = {"correct": 0, "incorrect": 0}

    if len(tasks_dict[task_list[0]]) <= 0:
        return

    for task_name, values in tasks_dict.items():
        preds, gts = [], []

        for original_obj, shuffled_obj in zip(values[0], values[1]):
            cos_sim = 1 - spatial.distance.cosine(original_obj.get_sentence_representation(), shuffled_obj.get_sentence_representation())
            if abs(cos_sim) >= 0.8:
                score_dict[task_name]["score"] += 1

            preds.append(shuffled_obj.get_pred_label())
            gts.append(shuffled_obj.get_ground_truth())

            label = label_dict[task_name][shuffled_obj.get_ground_truth()]
            if shuffled_obj.get_pred_label() == shuffled_obj.get_ground_truth():
                prediction_stats[task_name][label]["correct"] += 1
            else:
                prediction_stats[task_name][label]["incorrect"] += 1

        prediction_stats[task_name]["accuracy"] = round(accuracy_score(preds, gts), 4)
        prediction_stats[task_name]["matthew"] = round(matthews_corrcoef(preds, gts), 4)

    for task_name, values in score_dict.items():
        score_dict[task_name]["average"] = round(values["score"] / values["total"] * 100, 2)
        print(task_name + "\t" + str(score_dict[task_name]["score"]) + "\t" + str(score_dict[task_name]["total"]) + "\t" + str(score_dict[task_name]["average"]))

    for task_name, values in prediction_stats.items():
        for label, predictions in values.items():
            if label == "accuracy" or label == "matthew":
                continue

            print(task_name + "\t" + label + "\t" + str(predictions["correct"]) + "\t" + str(predictions["incorrect"]) + "\t" +
                  str(values["accuracy"]) + "\t" + str(values["matthew"]))


def generate_dev_set(task_name, task_id, labels, base_model, output_file, sub_folder=None):
    # Output will be a dictionary
    # output = {"label_1": [0,1,2,3,4], "label_2": [5,6,7,8,9]}

    # Extra finetuning
    # data_dir = base_dir + base_model + "/5th_400/" + task_name + "/evaluation_1st/" + (sub_folder + "/" if sub_folder is not None else "")

    # Normal finetuning
    bridge = "/evaluation_1st/" if base_model == "roberta-large" else "/evaluation_2nd/"
    data_dir = base_dir + base_model + "/" + task_name + bridge + (sub_folder + "/" if sub_folder is not None else "")

    path = data_dir + "original/" + "ml_objects.pickle"

    with open(path, "rb") as file_path:
        objs_original = pickle.load(file_path)

    if len(objs_original) <= 0:
        return

    correct_preds = {}
    for label in labels:
        correct_preds[label] = []

    for idx, obj in tqdm(enumerate(objs_original)):
        ground_truth = str(labels[int(obj.get_ground_truth())])
        prediction = str(labels[int(obj.get_pred_label())])

        if prediction == ground_truth:

            # --------------------------------------------------------
            # ThangPM: Added this block of code for Adversarial GLUE ONLY
            # --------------------------------------------------------
            sentence_to_be_shuffled = " ".join(obj.get_tokens()["text_a"])
            if task_id == "rte" or task_id == "mnli":
                sentence_to_be_shuffled = " ".join(obj.get_tokens()["text_b"])

            if not has_two_nouns_or_more(sentence_to_be_shuffled):
                continue
            # --------------------------------------------------------

            correct_preds[ground_truth].append(obj.get_guid())

    size_per_label = len(correct_preds[list(correct_preds.keys())[0]])
    for label, values in correct_preds.items():
        print(label + " --- " + str(len(values)))
        size_per_label = min(size_per_label, len(values))

    print(task_name)
    for label, values in correct_preds.items():
        correct_preds[label] = sorted(random.sample(values, size_per_label))
        print(label + " --- " + str(size_per_label))

    # print(correct_preds)

    if sub_folder is not None:
        task_id = sub_folder

    output_file.write(task_id + "||" + str(correct_preds) + "\n")

    return correct_preds


def run_all_experiments():

    # base_model = "roberta-base"
    # base_model = "albert-base-v2"
    base_model = "bert-base-uncased"

    # base_model = "roberta-base-cola"
    # base_model = "roberta-extra-finetune"

    # base_model = "roberta-large"

    modes = ["original", "shuffled", "shuffled_bigram", "shuffled_trigram"]
    tasks = {
             "cola": "CoLA",
             "mrpc": "MRPC",
             "sst2": "SST-2",
             "rte": "RTE",
             "qqp": "QQP",
             "qnli": "QNLI",
             "stsb": "STS-B",
             "mnli": "MNLI", # For ANLI task
             }

    seeds = [422]
    # seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # seeds = ["1st_42", "2nd_100", "3rd_200", "4th_300", "5th_400"]

    # Use for the function: generate_dev_set
    output_dir = base_dir + base_model + "/"
    # output_file = open(output_dir + "list_tasks_indices_rte_" + base_model + ".txt", "w")

    label_dict = {}
    for key, task_name in tasks.items():
        # if key == "mnli": # key == "stsb" or key == "cola" or
        #     continue
        # if key != "qqp" and key != "rte":
        #     continue
        # if key != "mnli":
        #     continue
        if key != "rte":
            continue

        print("Handling task name: " + task_name)

        dataset = nlp.load_dataset("glue", key)
        val_key = 'validation' if 'validation' in dataset else 'val'
        if key == "mnli":
            val_key = "validation_matched"
        dataset = dataset[val_key]
        if key != "stsb":
            labels = dataset.features['label'].names
            label_dict[task_name] = labels
        else:
            labels = []
            label_dict[task_name] = []

        # load_and_print_out_predictions(task_name, labels)

        # The order is different between processors/glue.py and huggingface datasets
        # The training is based on processors/glue.py so must follow its order.
        # if key == "mnli":
        #     sub_folders = ["r1", "r2", "r3"]  # For ANLI only
        #     labels = ["contradiction", "neutral", "entailment"] # For our ANLI finetuned RoBERTa-base model.
        #     # labels = ["entailment", "neutral", "contradiction"] # For ANLI authors' finetuned RoBERTa-large model.
        #     for sub_folder in sub_folders:
        #         generate_dev_set(task_name, key, labels, base_model, output_file, sub_folder=sub_folder)
        # else:
        #     generate_dev_set(task_name, key, labels, base_model, output_file)

        conf_mode = True
        if key == "mnli":
            sub_folders = ["r1", "r2", "r3"]  # For ANLI only
            labels = ["contradiction", "entailment", "neutral"]
            for sub_folder in sub_folders:
                for seed in seeds:
                    analyze_results_for_confidence_scores(task_name, labels, base_model, write_files=True, for_paper=False, seed=str(seed), sub_folder=sub_folder, conf_mode=conf_mode)
        else:
            for seed in seeds:
                analyze_results_for_confidence_scores(task_name, labels, base_model, write_files=True, for_paper=False, seed=str(seed), conf_mode=conf_mode)

        # analyze_statistics_for_only_stsb(task_name="STS-B", base_model=base_model)

    # output_file.close()

    # analyze_results_for_calibration(list(tasks.values()), label_dict, mode="shuffled")


def plot_stacked_bar(data, series_labels, category_labels=None,
                     show_values=False, value_format="{}", x_label=None, y_label=None,
                     colors=None, grid=True, reverse=False):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        color = colors[i] if colors is not None else None
        axes.append(plt.bar(ind, row_data, bottom=cum_size, label=series_labels[i], color=color))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)

    if x_label: plt.xlabel(x_label, fontsize=18)
    if y_label: plt.ylabel(y_label, fontsize=18)

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w/2, bar.get_y() + h/2, value_format.format(h), ha="center", va="center", fontsize=13, color="white")

    # plt.legend(loc=7)
    # plt.legend(bbox_to_anchor=(0.5, -0.3), loc="lower center", fontsize=12, bbox_transform=plt.gcf().transFigure)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=12)
    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=15)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

if __name__ == '__main__':

    run_all_experiments()

    # Unigram
    # data = [
    #     [35.39, 52.48, 72.06, 56.36, 84.03, 51.87],
    #     [11.69, 23.27, 10.87, 18.46, 12.5, 16.92],
    #     [52.92, 24.25, 17.07, 25.18, 3.47, 31.21],
    # ]

    # Bigram
    # data = [
    #     [42.21, 66.34, 79.3, 73.62, 85.42, 63.68],
    #     [11.36, 17.82, 8.78, 13.76, 9.72, 12.19],
    #     [46.43, 15.84, 11.92, 12.62, 4.86, 24.13]
    # ]

    # Trigram
    # data = [
    #     [44.48, 70.79, 81.21, 81.64, 89.58, 72.26],
    #     [13.31, 15.35, 8.01, 10.9, 6.94, 10.45],
    #     [42.21, 13.86, 10.78, 7.46, 3.48, 17.29]
    # ]
    #
    # plt.figure(figsize=(9.2, 6.4))
    # series_labels = ['s >= 0.9', '0.8 <= s < 0.9', 's < 0.8']
    # category_labels = ['CoLA', "MRPC", 'QQP', "QNLI", 'RTE', 'SST-2']
    #
    # plot_stacked_bar(
    #     data,
    #     series_labels,
    #     category_labels=category_labels,
    #     show_values=True,
    #     grid=False,
    #     value_format="{:.2f}",
    #     colors=['tab:blue', 'tab:orange', 'tab:green'],
    #     # x_label="Task",
    #     y_label="Percentage",
    # )
    #
    # plt.savefig('bar.pdf', bbox_inches="tight")
    # plt.show()