import pandas as pd
from typing import List

# import plotly.express as px


class Metrics:
    def __init__(self, labels: List[int]):
        self.labels = labels

    def get_tp_fp_fn_tn(self, scores: List[float]):
        tp, fp, fn, tn = 0, 0, 0, 0
        for i, label in enumerate(self.labels):
            test = scores[i]
            if label == 1 and test == 1:
                tp += 1
            elif label == 0 and test == 0:
                tn += 1
            elif label == 1 and test == 0:
                fn += 1
            elif label == 0 and test == 1:
                fp += 1
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn
        return tp, fp, fn, tn

    def get_pr_f1(self, scores):
        tp, fp, fn, tn = self.get_tp_fp_fn_tn(scores)
        self.precision = tp / (tp + fp)
        self.recall = tp / (tp + fn)
        self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        return self.precision, self.recall, self.f1

