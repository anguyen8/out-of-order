# Import libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

from adjustText import adjust_text

import numpy as np

# Creating dataset
np.random.seed(10)


def show_box_plot():
    # glue = [3.79, 1, 0.56, 0.3, 0.28, 0.23, 0.23, 0.23, 0.18, 0]
    # super_glue = [1.34, 0.8, 0.62, 0.23, 0.22, 0.2, 0.17, -0.1]
    # stanford = [0.44, 0.17, 0.17]
    # other = [0.65, 0.48, 0.29, 0.09]
    # box_plot_data = [glue, super_glue, stanford, other]
    # plt.boxplot(box_plot_data, patch_artist=True, labels=['GLUE','SuperGLUE','Stanford','Others'])

    coref = [1.34, 0]
    mc = [0.2, 0.09]
    nli = [0.56, 0.23, 0.23, 0.18, 0.17, 0.62, -0.1]
    qa = [0.44, 0.17, 0.8, 0.22, 0.17, 0.28]
    text_cls = [0.3, 0.65, 0.29]
    paraphrase = [1, 0.23]
    ner = [0.48]
    others = [3.79, 0.28, 0.23]

    box_plot_data = [coref, mc, nli, qa, text_cls, paraphrase, ner, others]
    plt.boxplot(box_plot_data, patch_artist=True, labels=['Coref', 'M_Choice', 'NLI', 'QA',
                                                          'Text_Cls', 'Paraphrase', 'NER', 'Others'])

    plt.show()


def show_scatter_plot():

    abbreviations = {"CoLA": "CLA", "WSC": "WSC", "RTE_G": "RTEG", "ReCoRD": "REC",
                     "RTE_SG": "RTES", "SemEval2014": "SE", "ATIS-NER": "NER", "COPA": "CPA",
                     "ATIS": "ATIS", "QQP": "QQP", "BoolQ": "BQ", "SST-2": "ST2", "MRPC": "MRPC",
                     "SQuAD-v1": "QA1", "WiC": "WIC", "MNLI-mm": "Mmm", "MNLI-m": "Mm", "QNLI": "QNLI",
                     "SNLI": "SNLI", "SWAG": "SWAG", "CB": "CB", "STS-B": "STS", "MultiRC": "MRC",
                     "WNLI": "WNLI", "SQuAD-v2": "QA2"}

    skipped_tasks = ["WNLI", "SemEval2014", "ATIS", "ATIS-NER"] # "RTE_G",

    import csv
    # file_name = 'table_Q1.csv'
    # file_name = 'table_Q1_albert.csv'
    # file_name = 'table_Q1_roberta.csv'
    # file_name = "table_Q1_roberta_subset.csv"
    file_name = "table_Q1_all_models_subset.csv"

    with open(file_name, mode='r') as infile:
        reader = csv.reader(infile)
        mydict = {rows[0]: [rows[1], rows[2]] + rows[9:15] + [rows[-6], rows[-3], rows[-2], rows[-1]] for rows in reader if rows[0] not in skipped_tasks}

    # print(mydict)
    mydict.pop("identifier", None)

    # Column Indices
    # task_type | benchmark |  original | shuffled_unigram | shuffled_bigram | shuffled_trigram |
    # baseline_dist | baseline_random | examples_dev_set | score_unigram | score_bigram | score_trigram

    shuffle_types = {"unigram": [0, 1, 2, 3, 6, 7, -3],
                     "bigram":  [0, 1, 2, 4, 6, 7, -2],
                     "trigram": [0, 1, 2, 5, 6, 7, -1]}

    merge_tasks = {}

    for shuffle_type, column_indices in shuffle_types.items():
        tasks = {}
        col_idx_0, col_idx_1, col_idx_2, col_idx_3, col_idx_4, col_idx_5, col_idx_6 = column_indices
        for key, values in mydict.items():
            if not key: continue

            task_type = values[col_idx_0]
            benchmark = values[col_idx_1]

            original = float(values[col_idx_2])
            shuffled = float(values[col_idx_3])
            baseline = float(values[col_idx_4]) if values[col_idx_4] != "" else float(values[col_idx_5])
            score = float(values[col_idx_6])

            # tasks[abbreviations[key]] = [original, shuffled, baseline, score, task_type, benchmark]
            tasks[key] = [original, shuffled, baseline, score, task_type, benchmark]

            if key not in merge_tasks:
                merge_tasks[key] = {"unigram": [], "bigram": [], "trigram": []}
            merge_tasks[key][shuffle_type] = [original, shuffled, baseline]

        # draw_scatter_plot_1(tasks)
        draw_scatter_plot_2(tasks, shuffle_type=shuffle_type)
        # draw_scatter_plot_3(tasks, shuffle_type=shuffle_type) # Use overall plot instead

    merged_scores = {}
    merged_types = {"unigram": [], "bigram": [], "trigram": []}
    for key, values in mydict.items():
        if not key: continue

        merged_scores[key] = [float(values[-3]), float(values[-2]), float(values[-1])]
        merged_types["unigram"].append(float(values[-3]))
        merged_types["bigram"].append(float(values[-2]))
        merged_types["trigram"].append(float(values[-1]))

    # draw_overall_plot_shuffled_vs_baseline(merge_tasks)
    draw_overall_plot(merged_scores, merged_types)


def draw_scatter_plot_1(tasks):
    tasks_names = []
    tasks_original = []
    tasks_shuffled = []

    for task_name, values in tasks.items():
        tasks_names.append(task_name)
        tasks_original.append(values[0])
        tasks_shuffled.append(values[1])

    fig, ax = plt.subplots()
    fig.suptitle("Gap Comparison between shuffle and baseline", fontsize=15)
    fig.set_size_inches(9.6, 9.6)

    for idx, task_name in enumerate(tasks_names):
        x, y = tasks_original[idx], tasks_shuffled[idx]
        ax.scatter(x, y, label=task_name, alpha=0.9, edgecolors='none')
        x_pos = 0
        if task_name == "mnli_m":
            x_pos = -20
        elif task_name == "mnli_mm":
            x_pos = 20
        ax.annotate(task_name, (x, y), textcoords="offset points", xytext=(x_pos, 5), ha='center', fontsize=9)

    ax.set_title("Original vs shuffled performance")
    # ax.legend(loc="lower center")
    ax.grid(True)

    plt.xlabel("Original", fontsize=12)
    plt.ylabel("Shuffled", fontsize=12)

    plt.xticks(np.arange(0, 110, 10))
    plt.yticks(np.arange(0, 110, 10))

    fig.savefig('plot_original_vs_shuffle.jpg')
    plt.show()


def draw_scatter_plot_2(tasks, shuffle_type="unigram"):
    tasks_names = []
    shuffled_gap = []
    baseline_gap = []

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
              "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    group_tasks = {"NLI": {},
                   "Q/A": {},
                   "Coreference Resolution": {},
                   "Sentiment Analysis": {}, # Text Classification
                   "Multiple Choice": {},
                   "Acceptability": {},
                   "Paraphrase": {},
                   "Sentence Similarity": {},
                   "Word Sense Disambiguation": {},
                   "NER": {},
                    }

    for task_name, values in tasks.items():
        tasks_names.append(task_name)
        shuffled_gap.append(values[0] - values[1])
        baseline_gap.append(values[0] - values[2])

        if "tasks" not in group_tasks[values[-2]]:
            group_tasks[values[-2]]["tasks"] = []
        if "shuffled" not in group_tasks[values[-2]]:
            group_tasks[values[-2]]["shuffled"] = []
        if "baseline" not in group_tasks[values[-2]]:
            group_tasks[values[-2]]["baseline"] = []

        group_tasks[values[-2]]["tasks"].append(task_name)
        group_tasks[values[-2]]["shuffled"].append(values[0] - values[1])
        group_tasks[values[-2]]["baseline"].append(values[0] - values[2])

    fig, ax = plt.subplots()
    fig.suptitle("Gap comparison between shuffle and baseline performance \n Shuffle Type: " + shuffle_type, fontsize=12)
    # fig.set_size_inches(9.6, 6.4)

    ax.set_xlim(xmin=-2)
    ax.set_ylim(ymin=-2)

    texts = []

    # Approach 1
    for task_type, task_values in group_tasks.items():
        if "tasks" not in task_values:
            continue

        sub_tasks = task_values["tasks"]
        x, y = task_values["shuffled"], task_values["baseline"]
        ax.scatter(x, y, label=task_type, alpha=0.9, edgecolors='none')

        for idx, task_name in enumerate(sub_tasks):
            # ax.annotate(task_name, (x[idx], y[idx]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9)
            texts.append(ax.text(x[idx], y[idx], task_name))

    # Approach 2: Visualize another way
    # for idx, task_name in enumerate(tasks_names):
    #     x, y = shuffled_gap[idx], baseline_gap[idx]
    #     ax.scatter(x, y, label=task_name, alpha=0.9, edgecolors='none')
    #     ax.annotate(task_name, (x, y), textcoords="offset points", xytext=(x_pos, y_pos), ha='center', fontsize=9)
    #     texts.append(ax.text(x, y, task_name))

    ax.legend(loc="best", ncol=1, fontsize=11)
    ax.grid(True)

    ax.set_xlabel("Original - Shuffle", fontsize=12)
    ax.set_ylabel("Original - Baseline", fontsize=12)

    plt.xticks(np.arange(0, 70, 5))
    plt.yticks(np.arange(0, 110, 10))

    # plt.xticks(np.arange(-10, 110, 5))
    # plt.yticks(np.arange(-10, 110, 5))

    adjust_text(texts, force_text=(2,2), arrowprops=dict(arrowstyle="-|>", color='dimgray', alpha=0.5))

    fig.savefig('plot_shuffle_baseline_gaps_' + shuffle_type + '.jpg')
    # plt.show()


def draw_scatter_plot_3(tasks, shuffle_type="unigram"):
    tasks_names = []
    tasks_scores = []

    for task_name, values in tasks.items():
        tasks_names.append(task_name)
        tasks_scores.append(values[3])

    fig, ax = plt.subplots()
    fig.suptitle("Scores of all tasks - Shuffle Type: " + shuffle_type, fontsize=15)
    fig.set_size_inches(9.6, 6.4)

    for idx, task_name in enumerate(tasks_names):
        x, y = idx, tasks_scores[idx]
        ax.scatter(x, y, label=task_name, alpha=0.9, edgecolors='none')
        x_pos = 0
        y_pos = 5 if idx % 2 == 0 else -12
        ax.annotate(task_name, (x, y), textcoords="offset points", xytext=(x_pos, y_pos), ha='center', fontsize=9)

    # ax.legend(loc="lower center")
    ax.grid(True)

    plt.xlabel("Task", fontsize=12)
    plt.ylabel("Score", fontsize=12)

    plt.xticks(np.arange(0, 25, 1))
    plt.yticks(np.arange(0, 5, 0.5))

    fig.savefig('plot_scores_' + shuffle_type + '.jpg')
    # plt.show()


def draw_overall_plot(merged_tasks, merged_types):

    markers = ['o', ',', '^']
    colors = ['dodgerblue', 'tomato', 'forestgreen']

    colors_full = ['brown', 'crimson', 'greenyellow', 'forestgreen', 'deepskyblue', 'blue', 'darkviolet', 'deeppink',
              'red', 'chocolate', 'darkorange', 'indigo', 'lawngreen', 'lime', 'darkcyan', 'deepskyblue',
              'tomato', 'goldenrod', 'lightseagreen', 'darkslategray', 'dodgerblue', 'grey', 'navy', 'magenta', 'pink',]

    fig, ax = plt.subplots()
    fig.suptitle("Scores of all tasks - Overall", fontsize=15)
    fig.set_size_inches(9.6, 6.4)

    task_names = list(merged_tasks.keys())

    # for idx, (task_name, values) in enumerate(merged_tasks.items()):
    #     x, y = np.array([idx] * len(values)), np.array(values)
    #     # color = list(mcolors.CSS4_COLORS.keys())[idx]
    #     color = colors_full[idx]
    #     ax.scatter(x[0], y[0], label="unigram", alpha=0.9, edgecolors='none', marker='.', color=color)
    #     ax.scatter(x[1], y[1], label="bigram", alpha=0.9, edgecolors='none', marker=',', color=color)
    #     ax.scatter(x[2], y[2], label="trigram", alpha=0.9, edgecolors='none', marker='^', color=color)
    #
    #     max_idx = np.argmax(y)
    #     ax.annotate(task_name, (x[max_idx], y[max_idx]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    for idx, (shuffle_type, values) in enumerate(merged_types.items()):
        x, y = np.arange(0, len(values)), np.array(values)
        ax.scatter(x, y, label=shuffle_type, alpha=0.9, edgecolors='none', marker=markers[idx], color=colors[idx])

        # if idx == 0:
        #     for i in range(len(y)):
        #         ax.annotate(task_names[i], (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    ax.legend(loc="upper right", ncol=1)
    ax.grid(True)

    plt.xlabel("Task", fontsize=12)
    plt.ylabel("Score", fontsize=12)

    plt.xticks(np.arange(0, len(task_names), 1), task_names, rotation=0)
    plt.yticks(np.arange(0, 1.1, 0.1))

    # print(mcolors.CSS4_COLORS)

    fig.savefig("plot_scores_overall.jpg")
    # plt.show()


def draw_overall_plot_shuffled_vs_baseline(merged_tasks):
    '''
    :param merged_tasks:
        # merge_tasks = {"CoLA": {"unigram": [original, shuffle, baseline],
        #                         "bigram" : [original, shuffle_bigram, baseline],
        #                         "trigram": [original, shuffle_trigram, baseline],},
        #                "WSC": {...},
        #                ...
        #                }
    :return:
    '''

    merged_types = {"unigram": [], "bigram": [], "trigram": []}
    tasks_names = merged_tasks.keys()

    for type in merged_types.keys():
        shuffled_gap = []
        baseline_gap = []

        for task_name, values in merged_tasks.items():
            original, shuffle, baseline = values[type]
            shuffled_gap.append(original - shuffle)
            baseline_gap.append(original - baseline)

        merged_types[type] = [shuffled_gap, baseline_gap]

    fig, ax = plt.subplots()
    fig.suptitle("Gap Comparison between shuffle and baseline - Overall", fontsize=15)
    # fig.set_size_inches(12.8, 12.8)
    fig.set_size_inches(12.8, 9.6)

    ax.set_xlim(xmin=-2)
    ax.set_ylim(ymin=-2)

    texts = []

    for shuffle_type, values in merged_types.items():
        x, y = np.array(values[0]), np.array(values[1])
        ax.scatter(x, y, label=shuffle_type, alpha=0.9, edgecolors='none')

        for idx, task_name in enumerate(tasks_names):
            texts.append(ax.text(x[idx], y[idx], task_name))

    ax.legend(loc="best", ncol=1)
    ax.grid(True)

    plt.xlabel("Original - Shuffled", fontsize=12)
    plt.ylabel("Original - Baseline", fontsize=12)

    plt.xticks(np.arange(0, 105, 10))
    plt.yticks(np.arange(0, 105, 10))

    adjust_text(texts, force_text=0.05, arrowprops=dict(arrowstyle="-|>", color='dimgray', alpha=0.5))

    fig.savefig('plot_shuffle_baseline_gaps_overall.jpg')
    # plt.show()


show_scatter_plot()
