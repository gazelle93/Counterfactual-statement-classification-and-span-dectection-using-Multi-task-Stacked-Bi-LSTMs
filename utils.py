import pandas as pd
import numpy as np
from text_processing import get_target, word_tokenization, get_nlp_pipeline


def get_processed_datasamples(datasamples, subtask):
    datasample_list = []

    if subtask == "1":
        for i in range(len(datasamples)):
            sentence = datasamples.iloc[i]['sentence']
            gold = datasamples.iloc[i]['gold_label']

            datasample_list.append([sentence, gold])

    else:
        for i in range(len(datasamples)):
            sentence = datasamples.iloc[i]['sentence']
            antecedent = datasamples.iloc[i]['antecedent']
            consequent = datasamples.iloc[i]['consequent']

            datasample_list.append([sentence, antecedent, consequent])

    return datasample_list

def load_datasamples(subtask):
    if subtask == "1":
        train_dataset = pd.read_csv("./data/Subtask-1/subtask1_train.csv")
        test_dataset = pd.read_csv("./data/Subtask-1/subtask1_test.csv")
    else:
        train_dataset = pd.read_csv("./data/Subtask-2/subtask2_train.csv")
        test_dataset = pd.read_csv("./data/Subtask-2/subtask2_test.csv")

    return train_dataset, test_dataset


def feature_to_idx(pos_datasamples, subtask):
    pos_to_ix = {}
    r_pos_to_ix = {}
    tag_to_ix = {}
    r_tag_to_ix = {}

    pos_list = []
    for pos_datasample in pos_datasamples:
        for pos in pos_datasample:
            if pos not in pos_list:
                pos_list.append(pos)

    for idx, pos in enumerate(pos_list):
            pos_to_ix[idx] = pos
            r_pos_to_ix[pos] = idx

    if subtask == "1":
        label_target = [0,1]
        for idx, tag in enumerate(label_target):
            tag_to_ix[idx] = tag
            r_tag_to_ix[tag] = idx

    else:
        sequence_span_target = ["A", "C", "O"]

        for idx, tag in enumerate(sequence_span_target):
            tag_to_ix[idx] = tag
            r_tag_to_ix[tag] = idx

    return pos_to_ix, r_pos_to_ix, tag_to_ix, r_tag_to_ix


def get_features(datasamples, nlp_pipeline, subtask, device):
    datasample_list = []
    selected_nlp_pipeline = get_nlp_pipeline(nlp_pipeline, device)

    if subtask == "1":
        for data in datasamples:
            cur_text = data[0][:-1]
            gold_label = data[1]
            tokens, pos_span_target = word_tokenization(cur_text, selected_nlp_pipeline, nlp_pipeline)
            datasample_list.append([tokens, pos_span_target, gold_label])
    else:
        for data in datasamples:
            cur_text = data[0][:-1]
            ant_span_text = data[1]
            con_span_text = data[2]
            tokens, pos_span_target, sequence_span_target = get_target(cur_text, ant_span_text, con_span_text, selected_nlp_pipeline, nlp_pipeline)
            datasample_list.append([tokens, pos_span_target, sequence_span_target])

    return np.array(datasample_list)


def get_confusion_matrix(gold_list, pred_list, _target):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for preds, golds in zip(pred_list, gold_list):
        if len(preds) == len(golds):
            for pred, gold in zip(preds, golds):
                if pred == gold == _target:
                    TP+=1
                elif pred == _target and pred != gold:
                    FP+=1
                elif gold == _target and pred != gold:
                    FN+=1
                else:
                    TN+=1
        else:
            for gold in golds:
                if gold == _target:
                    FN+=1
                else:
                    TN+=1

    return TP, FP, TN, FN

def get_preicion_recall_f1(TP, FP, FN):
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if precision == recall == 0:
        f1score = 0
    else:
        f1score = 2 * precision * recall / (precision + recall)

    return round(precision*100,2), round(recall*100,2), round(f1score*100,2)


def get_evaluation_result(gold_list, pred_list):
    TP_A, FP_A, TN_A, FN_A = get_confusion_matrix(gold_list, pred_list, "A")
    TP_C, FP_C, TN_C, FN_C = get_confusion_matrix(gold_list, pred_list, "C")

    precision_A, recall_A, f1score_A = get_preicion_recall_f1(TP_A, FP_A, FN_A)
    precision_C, recall_C, f1score_C = get_preicion_recall_f1(TP_C, FP_C, FN_C)

    precision = (precision_A + precision_C)/2
    recall = (recall_A + recall_C)/2
    if precision + recall == 0:
        f1score = 0
    else:
        f1score = 2 * precision * recall / (precision + recall)

    print("Subtask 2 result")
    print("Antecedent\tPrecision: {}, Recall: {}, F1-score: {}".format(precision_A, recall_A, f1score_A))
    print("Consequent\tPrecision: {}, Recall: {}, F1-score: {}".format(precision_C, recall_C, f1score_C))
    print("Macro-Avg\tPrecision: {}, Recall: {}, F1-score: {}".format(round(precision,2), round(recall,2), round(f1score,2)))
