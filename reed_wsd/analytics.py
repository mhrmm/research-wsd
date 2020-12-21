from sklearn import metrics
from reed_wsd.util import ABS
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


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
        precision, recall, _ = metrics.precision_recall_curve(self.y_true,
                                                              self.y_scores)
        precision = np.insert(precision, 0,
                              self.num_correct() / self.num_predictions(),
                              axis=0)
        recall = np.insert(recall, 0, 1.0, axis=0)
        auc = metrics.auc(recall, precision)
        return precision, recall, auc

    def roc_curve(self):
        if len(self.y_true) == 0 or len(self.y_scores) == 0:
            return None, None, None
        fpr, tpr, _ = metrics.roc_curve(self.y_true, self.y_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        return fpr, tpr, auc

    def risk_coverage_curve(self):
        # TODO: this function currently plots *unconditional* error rate
        #  against coverage
        if len(self.y_true) == 0 or len(self.y_scores) == 0:
            return None, None, None
        precision, _, thresholds = metrics.precision_recall_curve(self.y_true,
                                                                  self.y_scores)
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

    def plot_roc(self):
        fpr, tpr, auc = self.roc_curve()
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUROC = %0.2f' % auc)
        plt.legend(loc='lower right')
        axes = plt.gca()
        axes.set_ylim([-0.05, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    def plot_pr(self):
        precision, recall, auc = self.pr_curve()
        plt.title('Precision-Recall')
        plt.plot(recall, precision, 'b', label='AUPR = %0.2f' % auc)
        plt.legend(loc='lower right')
        axes = plt.gca()
        axes.set_ylim([-0.05, 1.05])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.show()

    @staticmethod
    def average_list_of_analytics(list_of_analytics):
        def sum_analytics_dicts(this, other):
            def elementwise_add(ls1, ls2):
                assert (len(ls1) == len(ls2))
                return [(ls1[i] + ls2[i]) for i in range(len(ls1))]

            def process_key(key):
                if key != 'prediction_by_class':
                    return this[key] + other[key]
                else:
                    return {key: elementwise_add(this[key], other[key])
                            for key in this.keys()}

            assert (this.keys() == other.keys())
            return {key: process_key(key) for key in this.keys()}

        def normalize_analytics_dict(d, divisor):
            def elementwise_div(ls):
                return [element / divisor for element in ls]

            def process_key(key):
                if key != 'prediction_by_class':
                    return d[key] / divisor
                else:
                    return {key: elementwise_div(d[key])
                            for key in d.keys()}

            return {key: process_key(key) for key in d.keys()}

        measurement_sum = reduce(sum_analytics_dicts, list_of_analytics)
        avg_measurement = normalize_analytics_dict(measurement_sum,
                                                   len(list_of_analytics))
        return avg_measurement


def plot_curves(*pycs):
    for i in range(len(pycs)):
        curve = pycs[i][0]
        label = pycs[i][1]
        label = label + "; aupy = {:.3f}".format(curve.aupy())
        curve.plot(label)
    plt.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()