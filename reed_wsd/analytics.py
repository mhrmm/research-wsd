from sklearn import metrics
from reed_wsd.util import ABS
import numpy as np


class Analytics:

    def __init__(self, predictions):
        self.y_true = [int(pred['pred'] == pred['gold']) for pred in predictions
                       if pred['pred'] != ABS]
        self.y_scores = [pred['confidence'] for pred in predictions
                         if pred['pred'] != ABS]
        self.avg_err_conf = 0
        self.avg_crr_conf = 0
        self.n_error = 0
        self.n_correct = 0
        self.n_published = 0
        self.n_preds = len(predictions)
        for result in predictions:
            prediction = result['pred']
            gold = result['gold']
            confidence = result['confidence']
            if prediction != ABS:
                self.n_published += 1
                if prediction == gold:
                    self.avg_crr_conf += confidence
                    self.n_correct += 1
                else:
                    self.avg_err_conf += confidence
                    self.n_error += 1

    def num_errors(self):
        return self.n_error

    def num_correct(self):
        return self.n_correct

    def num_published(self):
        return self.n_published

    def num_predictions(self):
        return self.n_preds

    def pr_curve(self):
        if len(self.y_true) == 0 or len(self.y_scores) == 0:
            return None, None, None
        precision, recall, _ = metrics.precision_recall_curve(self.y_true, self.y_scores)
        auc = metrics.auc(recall, precision)
        return precision, recall, auc

    def roc_curve(self):
        if len(self.y_true) == 0 or len(self.y_scores) == 0:
            return None, None, None
        fpr, tpr, _ = metrics.roc_curve(self.y_true, self.y_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        return fpr, tpr, auc

    def risk_coverage_curve(self):
        # this function plots unconditional error rate against coverage
        if len(self.y_true) == 0 or len(self.y_scores) == 0:
            return None, None, None
        precision, _, thresholds = metrics.precision_recall_curve(self.y_true, self.y_scores)
        y_scores = sorted(self.y_scores)
        coverage = []
        N = len(y_scores)
        j = 0
        for i, t in enumerate(thresholds):
            while j < len(y_scores) and y_scores[j] < t:
                j += 1
            coverage.append((N - j) / N)
        coverage += [0.]
        conditional_err = 1 - precision
        unconditional_err = conditional_err * coverage
        coverage = np.array(coverage)
        capacity = 1 - metrics.auc(coverage, unconditional_err)
        return coverage, unconditional_err, capacity

