from sklearn import metrics
from reed_wsd.util import ABS
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Ubuntu Condensed']


class EvaluationResult:

    def __init__(self, predictions, loss=None):
        self._loss = loss
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
        if len(self.y_true) == 0 or len(self.y_scores) == 0:
            self.fpr, self.tpr, self.auroc = None, None, None
            self.precision, self.recall, self.aupr = None, None, None
        else:
            self.fpr, self.tpr, _ = metrics.roc_curve(self.y_true, self.y_scores,
                                                      pos_label=1)
            self.auroc = metrics.auc(self.fpr, self.tpr)
            precision, recall, _ = metrics.precision_recall_curve(self.y_true,
                                                                  self.y_scores)
            self.precision = np.insert(precision, 0,
                                       self.num_correct() / self.num_predictions(),
                                       axis=0)
            self.recall = np.insert(recall, 0, 1.0, axis=0)
            self.aupr = metrics.auc(self.recall, self.precision)

    def loss(self):
        return self._loss

    def num_errors(self):
        return self.n_error

    def num_correct(self):
        return self.n_correct

    def num_published(self):
        return self.n_published

    def num_predictions(self):
        return self.n_preds

    def pr_curve(self):
        return self.precision, self.recall, self.aupr

    def roc_curve(self):
        return self.fpr, self.tpr, self.auroc

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

    def as_dict(self):
        _, _, auroc = self.roc_curve()
        _, _, aupr = self.pr_curve()
        _, _, capacity = self.risk_coverage_curve()
        return {'avg_err_conf': (self.avg_err_conf / self.n_error
                                 if self.n_error > 0 else 0),
                'avg_crr_conf': (self.avg_crr_conf / self.n_correct
                                 if self.n_correct > 0 else 0),
                'auroc': auroc,
                'aupr': aupr,
                'capacity': capacity,
                'precision': (self.n_correct / self.n_published
                              if self.n_published > 0 else 0),
                'coverage': (self.n_published / self.n_preds
                             if self.n_preds > 0 else 0)}

    def __str__(self):
        d = self.as_dict()
        return '  ' + '\n  '.join(['{}: {}'.format(key, d[key]) for key in d])


class EpochResult:

    def __init__(self, train_loss, validation_result):
        self.train_loss = train_loss
        self.validation_result = validation_result

    def get_train_loss(self):
        return self.train_loss


class Analytics:

    def __init__(self, epoch_results):
        self.epoch_results = epoch_results

    def show_training_dashboard(self):
        fig, (ax1, ax2) = plt.subplots(2, sharex='all')
        fig.suptitle('Training Dashboard', fontsize='18')
        indexed_results = [(i+1, r) for (i, r) in enumerate(self.epoch_results)]
        x_axis = [i for (i, _) in indexed_results]
        train_losses = [r.get_train_loss() for (_, r) in indexed_results]
        valid_losses = [r.validation_result.loss() for (_, r) in indexed_results]
        valid_aurocs = [r.validation_result.auroc for (_, r) in indexed_results]
        valid_auprs = [r.validation_result.aupr for (_, r) in indexed_results]
        ax1.plot(x_axis, train_losses, 'b', label='train loss')
        ax1.plot(x_axis, valid_losses, 'r', label='valid loss')
        ax1.set(ylabel='loss')
        ax2.plot(x_axis, valid_aurocs, 'g', label='valid auroc')
        ax2.plot(x_axis, valid_auprs, 'orange', label='valid aupr')
        ax1.legend(loc='upper right')
        ax2.legend(loc='lower right')
        ax2.set(ylabel='metric')
        plt.xlabel('epoch')
        plt.show()

    @staticmethod
    def average_list_of_results(list_of_results):
        def sum_result_dicts(this, other):
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

        def normalize_result_dict(d, divisor):
            def elementwise_div(ls):
                return [element / divisor for element in ls]

            def process_key(key):
                if key != 'prediction_by_class':
                    return d[key] / divisor
                else:
                    return {key: elementwise_div(d[key])
                            for key in d.keys()}

            return {key: process_key(key) for key in d.keys()}

        result_sum = reduce(sum_result_dicts, list_of_results)
        avg_result = normalize_result_dict(result_sum,
                                           len(list_of_results))
        return avg_result


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