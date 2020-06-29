import unittest
import torch
from torch import tensor
from allwords.evaluate import predict, accuracy, yielde, ABSTAIN
from allwords.evaluate import py_curve
from allwords.evaluate import apply_zone_masks

class TestEvaluate(unittest.TestCase):
    
    def test_predict(self):
        t = tensor([0.35, 0.2, 0.1, 0.05, 0.3])
        assert(predict(t) == 0)
        t = tensor([0.1, 0.35, 0.2, 0.05, 0.3])
        assert(predict(t) == 1)
        t = tensor([0.1, 0.2, 0.35, 0.05, 0.3])
        assert(predict(t) == 2)
        t = tensor([0.1, 0.2, 0.05, 0.35, 0.3])
        assert(predict(t) == 3)
        t = tensor([0.1, 0.2, 0.3, 0.05, 0.35])
        assert(predict(t) == 4)
        
    def test_accuracy1(self):
        predicted = [3,6,5,1,2,2,2,4,1,3]
        gold = [3,6,5,2,2,2,1,4,7,3]
        assert(accuracy(predicted, gold) == (7, 10))

    def test_accuracy2(self):
        predicted = [3,6,5,ABSTAIN,2,2,2,4,ABSTAIN,3]
        gold = [3,6,5,2,2,2,1,4,7,3]
        assert(accuracy(predicted, gold) == (7, 8))

    def test_accuracy3(self):
        predicted = [3,ABSTAIN,5,ABSTAIN,2,2,2,4,ABSTAIN,3]
        gold = [3,6,5,2,2,2,1,4,7,3]
        assert(accuracy(predicted, gold) == (6, 7))

    def test_yield(self):
        predicted = [3,6,5,1,2,2,2,4,1,3]
        gold = [3,6,5,2,2,2,1,4,7,3]
        assert(yielde(predicted, gold) == (7, 10))
 
    def test_yield2(self):
        predicted = [3,6,5,ABSTAIN,2,2,2,4,ABSTAIN,3]
        gold = [3,6,5,2,2,2,1,4,7,3]
        assert(yielde(predicted, gold) == (7, 10))

    def test_yield3(self):
        predicted = [3,ABSTAIN,5,ABSTAIN,2,2,2,4,ABSTAIN,3]
        gold = [3,6,5,2,2,2,1,4,7,3]
        assert(yielde(predicted, gold) == (6, 10))
    
    def test_py_curve(self):
        preds = [3,6,5,1,2]
        gold = [3,6,5,2,2]
        confs = [0.1 * i for i in range(5)]
        expected = {0.0: [0.8, 0.8], 0.1: [0.75, 0.6], 
                    0.2: [0.667, 0.4], 
                    0.3: [0.5, 0.2], 0.4: [1.0, 0.2]} 
        for (key, (p, r)) in py_curve(preds, gold, confs).items():
            key = round(key*10)/10
            (other_p, other_r) = expected[key]
            assert (round(p*1000)/1000 == other_p)
            assert (round(r*1000)/1000 == other_r)

    
    def test_apply_zone_masks(self):
        t = tensor([[0.2, 0.2, 0.1, 0.05, 0.45],
                    [0.2, 0.4, 0.1, 0.2, 0.1]])      
        zones = [(0, 3), (2, 5)]
        expected = tensor([[0.4000, 0.4000, 0.2000, 0.0000, 0.0000],
                           [0.0000, 0.0000, 0.2500, 0.5000, 0.2500]])
        assert torch.all(torch.eq(expected, apply_zone_masks(t, zones)))
        
        
if __name__ == "__main__":
	unittest.main()