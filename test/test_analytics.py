import unittest
from reed_wsd.analytics import EvaluationResult
import numpy as np


def compare(a1, a2, num_decimal_places=4):
    comparison = a1.round(num_decimal_places) == a2.round(num_decimal_places)
    return comparison.all()


def approx(x, y, num_decimal_places=4):
    return round(x, num_decimal_places) == round(y, num_decimal_places)


class TestEvaluationResult(unittest.TestCase):
    
    def setUp(self):
        self.preds1 = [{'gold': 9, 'pred': 4, 'confidence': 0.1},
                       {'gold': 5, 'pred': 5, 'confidence': 0.3},
                       {'gold': 7, 'pred': 1, 'confidence': 0.5},
                       {'gold': 2, 'pred': 2, 'confidence': 0.7},
                       {'gold': 3, 'pred': 3, 'confidence': 0.9}]
        
        self.preds2 = [{'pred': 1, 'gold': 1, 'confidence': 0.1},
                       {'pred': 8, 'gold': 7, 'confidence': 0.1},
                       {'pred': 4, 'gold': 9, 'confidence': 0.4},
                       {'pred': 6, 'gold': 6, 'confidence': 0.35},
                       {'pred': 3, 'gold': 3, 'confidence': 0.8}]

    def test_pr_curve(self):
        result = EvaluationResult(self.preds1)
        precision, recall, auc = result.pr_curve()
        assert compare(precision, np.array([0.6, 0.75, 0.66666667, 1., 1., 1.]))
        assert compare(recall, np.array([1., 1., 0.66666667, 0.66666667,
                                         0.33333333, 0.]))
        assert approx(auc, 0.9027777777777777)
        
    def test_roc_curve(self):
        result = EvaluationResult(self.preds1)
        fpr, tpr, auc = result.roc_curve()
        assert compare(fpr, np.array([0., 0., 0., 0.5, 0.5, 1. ]))
        assert compare(tpr, np.array([0., 0.33333333, 0.66666667, 
                                      0.66666667, 1., 1. ]))
        assert approx(auc, 0.8333333333333333)

    def test_risk_coverage_curve(self):
        result = EvaluationResult(self.preds1)
        coverage, risk, capacity = result.risk_coverage_curve()
        expected_risk = np.array([0.2, 0.2, 0., 0., 0.])
        expected_coverage = np.array([0.8, 0.6, 0.4, 0.2, 0.])
        assert compare(expected_risk, risk)
        assert compare(expected_coverage, coverage)
        assert approx(capacity, 0.94)


if __name__ == "__main__":
    unittest.main()
