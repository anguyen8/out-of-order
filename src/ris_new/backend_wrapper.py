
import re
import operator
import datetime
import random

import torch
import numpy as np
from numpy import linalg as LA

import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt

from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from scipy.stats import spearmanr
from scipy.special import softmax

from matplotlib.backends.backend_pdf import PdfPages
from pdflatex import PDFLaTeX

from lime import lime_text
from lime.lime_text import LimeTextExplainer

from transformers import BertConfig, BertTokenizer, XLNetConfig, XLNetTokenizer, WEIGHTS_NAME
from src.atae_bert_model_wrapper import ATAE_BERT_Model_Wrapper
from src.atae_lstm_model_wrapper import ATAE_LSTM_Model_Wrapper
from src.utils import *
from src.attribution_visualization import AttributionVisualization
from src.ris_attribution import RIS_Analyzer, RIS_Evaluation
from src.evaluation import QuantitativeEvaluation
from atae_bert import ATAE_BERT

from LAMA.lama.modules import build_model_by_name

import json
import ast

ROUND_NUMBER = 3


class Backend_Wrapper(object):

    def __init__(self):
        # Prepare a classifier for LIME to generate explanations
        self.model_wrapper_bert = ATAE_BERT_Model_Wrapper()
        # self.model_wrapper_bert = None
        self.model_wrapper_lstm = ATAE_LSTM_Model_Wrapper()

        self.model_wrapper = self.model_wrapper_lstm

        # Initialize BERT model for inpainting
        self.inpaint_model = {}
        for lm in self.model_wrapper.args.models_names:
            self.inpaint_model[lm] = build_model_by_name(lm, self.model_wrapper.args)

    def get_model_wrapper(self):
        return self.model_wrapper

    def lime_explain(self, explainer, model_wrapper, class_names, text_instance):

        exp = explainer.explain_instance(text_instance=text_instance,
                                         classifier_fn=model_wrapper.predict_proba,
                                         top_labels=len(class_names),
                                         num_features=len(text_instance.split(' ')),
                                         num_samples=1000)

        prob = model_wrapper.predict_proba([text_instance])
        pred_label = np.argmax(prob)

        print('LIME Probability =', prob)

        return exp, pred_label

    def occlusion_explain(self, model_wrapper, sentence, replacement="<unk>"):

        tokens = sentence.split(" ")

        occlusion_scores = []
        replaced_sentences = []

        for i, token in enumerate(tokens):
            # Replace i-th token by the given replacement which could be <unk> or empty
            if replacement != "":
                replaced_sentence = " ".join(tokens[:i] + [replacement] + tokens[i + 1:])
            else:
                replaced_sentence = " ".join(tokens[:i] + tokens[i + 1:])

            replaced_sentences.append(replaced_sentence)

        # Get predictions for the original sentence and replaced sentences
        all_logits, _ = model_wrapper.predict_proba([sentence] + replaced_sentences, is_lime=False, grad_based=False, debug_mode=False)

        predict_label = np.argmax(all_logits[0])

        # Compute attribution scores
        for logits in all_logits[1:]:
            scores = all_logits[0] - logits
            occlusion_scores.append(scores)

        occlusion_scores = [score[predict_label] for score in occlusion_scores]

        return predict_label, occlusion_scores

    def deepLIFT_explain(self):
        pass

    '''
        Frontend analysers: analyzer_choices = [(1, "RIS"), (2, "ATT"), (3, "GRAD"), (4, "LIME"), (5, "OCCL_UNK"), (6, "OCCL_EMPTY")]
        Notes for analyzers:
        - 1: RIS
        - 2: ATT
        - 3: GRAD
        - 4: LIME
        - 5: OCCL_UNK
        - 6: OCCL_EMPTY
    '''

    def run_analyzers(self, top_N, ris_type, sentence=None, aspect=None, analyzers=None):

        class_names, dataset_size = self.model_wrapper.get_dataset_info()
        self.model_wrapper.set_aspect(aspect)

        # Create an LIME explainer object
        explainer = LimeTextExplainer(class_names=class_names)

        # RIS
        ris_analyzer = RIS_Analyzer(self.model_wrapper, self.inpaint_model, top_N, ris_type)

        attribution_viz_list, ris_explanation_list = [], []

        tokenized_sentence = sentence
        tokens = sentence.split(" ")

        attribution_scores, att_weights, norm, LIME_scores, occlusion_unk_scores, occlusion_empty_scores = [], [], [], [], [], []
        predicted_label_ris, pred_label_att_grad, pred_label_lime, pred_label_occl_unk, pred_label_occl_empty = -1, -1, -1, -1, -1

        # ------------------------------------------------------------------------------------------------------
        #   Prediction ONLY
        # ------------------------------------------------------------------------------------------------------
        all_logits, _ = self.model_wrapper.predict_proba([tokenized_sentence], is_lime=False, grad_based=False, debug_mode=False)
        logits = all_logits[0]
        pred_label = np.argmax(logits)
        print("Logits: " + str(logits))
        # ------------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------------
        # Occlusion Attribution Method -> Use <unk> and empty for the replacements
        # ------------------------------------------------------------------------------------------------------
        if '5' in analyzers:
            pred_label_occl_unk, occlusion_unk_scores = self.occlusion_explain(self.model_wrapper, tokenized_sentence, replacement="<unk>")

            print("Predicted label Occlusion UNK: " + str(pred_label_occl_unk))
            print("Occlusion UNK scores: " + str(occlusion_unk_scores))

        if '6' in analyzers:
            pred_label_occl_empty, occlusion_empty_scores = self.occlusion_explain(self.model_wrapper, tokenized_sentence, replacement="")

            print("Predicted label Occlusion EMPTY: " + str(pred_label_occl_empty))
            print("Occlusion EMPTY scores: " + str(occlusion_empty_scores))
        # ------------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------------
        # Attention + Gradient-base Attribution Method
        # ------------------------------------------------------------------------------------------------------
        if isinstance(self.model_wrapper, ATAE_LSTM_Model_Wrapper):
            _, all_att_weights, all_grads = self.model_wrapper.predict_proba([tokenized_sentence], is_lime=False, grad_based=True, debug_mode=False)

            # Since we handle one sentence at the same time --> Get the first element
            att_weights, grads = all_att_weights[0], all_grads[0]

            norm = LA.norm(grads, axis=1)
            norm /= max(norm)

            att_weights = att_weights[:len(tokens)]
            norm = norm[:len(tokens)]

            print("Attention: " + str(att_weights))
            print("Grad: " + str(norm))

            if '2' not in analyzers:
                att_weights = np.array([])
            if '3' not in analyzers:
                norm = np.array([])

            pred_label_att_grad = np.argmax(logits)
            print("Predicted label Att + Grad: " + str(pred_label_att_grad))
        # ------------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------------
        # Run LIME and save figures as LIME explanations
        # ------------------------------------------------------------------------------------------------------
        if '4' in analyzers:
            lime_explanations, pred_label_lime = self.lime_explain(explainer, self.model_wrapper, class_names, tokenized_sentence)
            print(lime_explanations.as_list(label=pred_label_lime))

            LIME_scores = []
            for token in tokens:
                t_score = 0
                for word, score in lime_explanations.as_list(label=pred_label_lime):
                    if token.lower() == word.lower():
                        t_score = score

                LIME_scores.append(t_score)

            print("LIME scores: " + str(LIME_scores))
        # ------------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------------
        # RIS
        # ------------------------------------------------------------------------------------------------------
        if '1' in analyzers:
            predicted_label_ris, attribution_scores = ris_analyzer.run_vis_attribution_method(tokenized_sentence, -1, aspect, ground_truth=None)
            ris_explanation_list = ris_analyzer.get_ris_explanation_list()

            print("RIS Predicted label: " + predicted_label_ris)
            print("RIS attribution scores: " + str(attribution_scores))
        # ------------------------------------------------------------------------------------------------------

        # NOTES
        # predicted_label_ris in ["positive", "negative", "neutral", "conflict"]
        # but others in [0, 1, 2, 3]
        # pred_label_att_grad is an original prediction label since it's label of the original sentence
        # if pred_label_occl_unk != pred_label_occl_empty:
        #     print("Sentence: " + sentence + " --- Case 1")
        # if pred_label_occl_empty != pred_label_att_grad:
        #     print("Sentence: " + sentence + " --- Case 2")
        # if pred_label_att_grad != pred_label_lime:
        #     print("Sentence: " + sentence + " --- Case 3")
        # if pred_label_lime != (self.model_wrapper.get_sentiment_vocab().stoi[predicted_label_ris] - 1):
        #     print("Sentence: " + sentence + " --- Case 4")

        # Store results for visualization
        # pred_label + 1 ti get predicted sentiment since the first item is <unk> so we have to shift right one
        attribution_viz = AttributionVisualization(tokens, aspect,
                                                   self.model_wrapper.get_sentiment_vocab(stoi=False)[pred_label + 1],
                                                   logits, None, attribution_scores, att_weights, norm, LIME_scores,
                                                   occlusion_unk_scores, occlusion_empty_scores)

        if ris_explanation_list is not None and len(ris_explanation_list) > 0:
            attribution_viz.set_ris_explanation_list(ris_explanation_list)

        attribution_viz_list.append(attribution_viz)

        # ------------------------------------------
        # ------------------------------------------

        # Visualize results
        # save_attribution_viz_baselines("../attribution_computation/" + ris_type + "_" + str(top_N) + "/", attribution_viz_list)
        # visualize_results(attribution_viz_list, ris_type, top_N)

        return attribution_viz_list

    def handle_request(self, model_type, sentence, aspect, analyzers, ris_type, top_N):
        start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # ---------------------------------------------------------------------
        #   HANDLE REQUESTS FROM USERS
        # ---------------------------------------------------------------------

        if model_type == "atae_lstm":
            self.model_wrapper = self.model_wrapper_lstm
        else:
            self.model_wrapper = self.model_wrapper_bert

        attribution_viz_list = self.run_analyzers(top_N=top_N, ris_type=ris_type, sentence=sentence, aspect=aspect, analyzers=analyzers)

        quant_eval = QuantitativeEvaluation(self.model_wrapper, attribution_viz_list)
        stored_path, _, _, _, _, _, _, _ = quant_eval.deletion_metric(ris_type + "_" + str(top_N))

        eval_results = load_evaluation_computation_results(stored_path)
        for eval_result in eval_results:
            if '1' in analyzers:
                eval_result['ris'] = eval_result['ris'].to_dict()
            else:
                eval_result['ris'] = {}
            if '2' in analyzers and eval_result['att'] is not None:
                eval_result['att'] = eval_result['att'].to_dict()
            else:
                eval_result['att'] = {}
            if '3' in analyzers and eval_result['grad'] is not None:
                eval_result['grad'] = eval_result['grad'].to_dict()
            else:
                eval_result['grad'] = {}
            if '4' in analyzers:
                eval_result['lime'] = eval_result['lime'].to_dict()
            else:
                eval_result['lime'] = {}
            if '5' in analyzers:
                eval_result['occlusion_unk'] = eval_result['occlusion_unk'].to_dict()
            else:
                eval_result['occlusion_unk'] = {}
            if '6' in analyzers:
                eval_result['occlusion_empty'] = eval_result['occlusion_empty'].to_dict()
            else:
                eval_result['occlusion_empty'] = {}

            # If at least one analyzer is selected --> Show the baseline for deletion method
            if len(analyzers) > 0:
                eval_result['baseline_l2r'] = eval_result['baseline_l2r'].to_dict()
            else:
                eval_result['baseline_l2r'] = {}

        # debug(eval_results, analyzers)

        message = json.dumps({"attribution_viz_list": [obj.to_dict() for obj in attribution_viz_list], "eval_results": eval_results})

        end_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        print("*** Start time: " + start_time + " ***")
        print("*** End time: " + end_time + " ***")

        # print("@@@@@Messages@@@@@" + message)
        # ---------------------------------------------------------------------

        return message

    def debug(self, eval_results, analyzers):
        for eval_result in eval_results:
            ris = eval_result["ris"]
            att = eval_result["att"]
            grad = eval_result["grad"]
            lime = eval_result["lime"]
            occl_unk = eval_result["occlusion_unk"]
            occl_empty = eval_result["occlusion_empty"]

            if '1' in analyzers:
                ris["pred_scores"] = ast.literal_eval(ris["pred_scores"])
                ris["auc_score"] = round(float(ris["auc_score"]), ROUND_NUMBER)
                ris["auc_score_norm"] = round(float(ris["auc_score_norm"]), ROUND_NUMBER)
            else:
                ris["pred_scores"] = []
                ris["auc_score"] = []
                ris["auc_score_norm"] = []

            if '2' in analyzers:
                att["pred_scores"] = ast.literal_eval(att["pred_scores"])
                att["auc_score"] = round(float(att["auc_score"]), ROUND_NUMBER)
                att["auc_score_norm"] = round(float(att["auc_score_norm"]), ROUND_NUMBER)
            else:
                att["pred_scores"] = []
                att["auc_score"] = []
                att["auc_score_norm"] = []

            if '3' in analyzers:
                grad["pred_scores"] = ast.literal_eval(grad["pred_scores"])
                grad["auc_score"] = round(float(grad["auc_score"]), ROUND_NUMBER)
                grad["auc_score_norm"] = round(float(grad["auc_score_norm"]), ROUND_NUMBER)
            else:
                grad["pred_scores"] = []
                grad["auc_score"] = []
                grad["auc_score_norm"] = []

            if '4' in analyzers:
                lime["pred_scores"] = ast.literal_eval(lime["pred_scores"])
                lime["auc_score"] = round(float(lime["auc_score"]), ROUND_NUMBER)
                lime["auc_score_norm"] = round(float(lime["auc_score_norm"]), ROUND_NUMBER)
            else:
                lime["pred_scores"] = []
                lime["auc_score"] = []
                lime["auc_score_norm"] = []

            if '5' in analyzers:
                occl_unk["pred_scores"] = ast.literal_eval(occl_unk["pred_scores"])
                occl_unk["auc_score"] = round(float(occl_unk["auc_score"]), ROUND_NUMBER)
                occl_unk["auc_score_norm"] = round(float(occl_unk["auc_score_norm"]), ROUND_NUMBER)
            else:
                occl_unk["pred_scores"] = []
                occl_unk["auc_score"] = []
                occl_unk["auc_score_norm"] = []

            if '6' in analyzers:
                occl_empty["pred_scores"] = ast.literal_eval(occl_empty["pred_scores"])
                occl_empty["auc_score"] = round(float(occl_empty["auc_score"]), ROUND_NUMBER)
                occl_empty["auc_score_norm"] = round(float(occl_empty["auc_score_norm"]), ROUND_NUMBER)
            else:
                occl_empty["pred_scores"] = []
                occl_empty["auc_score"] = []
                occl_empty["auc_score_norm"] = []

            print("RIS: " + str(ris['removed_sentences']))
            print("ATT: " + str(att['removed_sentences']))
            print("GRAD: " + str(grad['removed_sentences']))
            print("LIME: " + str(lime['removed_sentences']))
            print("OCCL_UNK: " + str(occl_unk['removed_sentences']))
            print("OCCL_EMPTY: " + str(occl_empty['removed_sentences']))

            fig = self.visualize_auc_figure(ris, att, grad, lime, occl_unk, occl_empty)
            fig.savefig("debug.jpg", format='jpg', dpi=300)

    def visualize_auc_figure(self, ris_eval_result, att_eval_result, grad_eval_result, lime_eval_result, occl_unk_eval_result, occl_empty_eval_result, show_plot=False):

        # Usage Note: https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.subplots.html
        sentence_length = 0
        flags = [False] * 6

        fig, ax = plt.subplots()

        if len(ris_eval_result["pred_scores"]) != 0:
            sentence_length = len(ris_eval_result["pred_scores"]) - 1  # Orginal sentence does not count
            x = np.arange(1, sentence_length + 1, 1)
            y_ris = np.array(ris_eval_result["pred_scores"][1:])
            ax.plot(x, y_ris, alpha=0.5)
            flags[0] = True
        if len(att_eval_result["pred_scores"]) != 0:
            sentence_length = len(att_eval_result["pred_scores"]) - 1  # Orginal sentence does not count
            x = np.arange(1, sentence_length + 1, 1)
            y_att = np.array(att_eval_result["pred_scores"][1:])
            ax.plot(x, y_att, alpha=0.5)
            flags[1] = True
        if len(grad_eval_result["pred_scores"]) != 0:
            sentence_length = len(grad_eval_result["pred_scores"]) - 1  # Orginal sentence does not count
            x = np.arange(1, sentence_length + 1, 1)
            y_grad = np.array(grad_eval_result["pred_scores"][1:])
            ax.plot(x, y_grad, alpha=0.5)
            flags[2] = True
        if len(lime_eval_result["pred_scores"]) != 0:
            sentence_length = len(lime_eval_result["pred_scores"]) - 1  # Orginal sentence does not count
            x = np.arange(1, sentence_length + 1, 1)
            y_lime = np.array(lime_eval_result["pred_scores"][1:])
            ax.plot(x, y_lime, alpha=0.5)
            flags[3] = True
        if len(occl_unk_eval_result["pred_scores"]) != 0:
            sentence_length = len(occl_unk_eval_result["pred_scores"]) - 1  # Orginal sentence does not count
            x = np.arange(1, sentence_length + 1, 1)
            y_occ_unk = np.array(occl_unk_eval_result["pred_scores"][1:])
            ax.plot(x, y_occ_unk, alpha=0.5)
            flags[4] = True
        if len(occl_empty_eval_result["pred_scores"]) != 0:
            sentence_length = len(occl_empty_eval_result["pred_scores"]) - 1  # Orginal sentence does not count
            x = np.arange(1, sentence_length + 1, 1)
            y_occ_emp = np.array(occl_empty_eval_result["pred_scores"][1:])
            ax.plot(x, y_occ_emp, alpha=0.5)
            flags[5] = True

        ax.set_title('AUC Metric')
        ax.xaxis.set_label_text('# removal words')
        ax.yaxis.set_label_text('Probability')

        legends = ['RIS', 'ATT', 'GRAD', 'LIME', 'OCC_UNK', 'OCC_EMP']
        legends_selected = [legend for idx, legend in enumerate(legends) if flags[idx]]

        ax.legend(legends_selected)

        if show_plot:
            plt.show()

        return fig


