import numpy as np
import scipy.stats
from scipy import stats

results = {
    "roar": {
        "10": {
            "loo_empty": [0.7511, 0.7339, 0.7420, 0.7500, 0.7523],
            "im": [0.7466, 0.7546, 0.7729, 0.7626, 0.7741],
            "baseline": [0.8979, 0.8968, 0.8888, 0.8922, 0.8853],
        },
        "20": {
            "loo_empty": [0.6881, 0.6778, 0.6800, 0.6869, 0.7144],
            "im": [0.6892, 0.6995, 0.7053, 0.7064, 0.703],
            "baseline": [0.8796, 0.8761, 0.8796, 0.8761, 0.8761],
        },
        "30": {
            "loo_empty": [0.6823, 0.6663, 0.6823, 0.6766, 0.6869],
            "im": [0.656, 0.6422, 0.6904, 0.6606, 0.6778],
            "baseline": [0.8647, 0.8521, 0.8544, 0.8521, 0.8578],
        }
    },
    "roar_bert": {
        "10": {
            "loo_empty": [0.7603, 0.7729, 0.7638, 0.7729, 0.7695],
            "im": [0.7603, 0.7729, 0.7787, 0.7718, 0.7844],
            "baseline": [0.8922, 0.8922, 0.8979, 0.8876, 0.8991],
        },
        "20": {
            "loo_empty": [0.7167, 0.7259, 0.7076, 0.7225, 0.7248],
            "im": [0.7133, 0.7007, 0.7385, 0.7225, 0.703],
            "baseline": [0.8842, 0.8796, 0.8784, 0.8842, 0.8853],
        },
        "30": {
            "loo_empty": [0.6835, 0.6858, 0.6594, 0.6686, 0.6835],
            "im": [0.6869, 0.6789, 0.672, 0.6835, 0.6628],
            "baseline": [0.8521, 0.8486, 0.8601, 0.8486, 0.8509],
        }
    }
}

for eval_name, removal_rates in results.items():
    for key in removal_rates.keys():
        loo_empty = removal_rates[key]["loo_empty"]
        im = removal_rates[key]["im"]
        baseline = removal_rates[key]["baseline"]

        loo_vs_im = stats.ttest_ind(np.array(loo_empty), np.array(im), equal_var=False)
        loo_vs_baseline = stats.ttest_ind(np.array(loo_empty), np.array(baseline), equal_var=False)
        im_vs_baseline = stats.ttest_ind(np.array(im), np.array(baseline), equal_var=False)

        print("Evaluation method: {} with {}%".format(eval_name, key))
        print("ttest loo vs im:\t{}\t{}".format(round(loo_vs_im.statistic, 4), round(loo_vs_im.pvalue, 4)))
        print("ttest loo vs baseline:\t{}\t{}".format(round(loo_vs_baseline.statistic, 4), round(loo_vs_baseline.pvalue, 4)))
        print("ttest im vs baseline:\t{}\t{}".format(round(im_vs_baseline.statistic, 4), round(im_vs_baseline.pvalue, 4)))









