import unittest
import os
from os.path import join
import sys
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, join(file_dir, ".."))
import torch
from reed_wsd.mnist.loss import ConfidenceLoss1
from reed_wsd.loss import PairwiseConfidenceLoss

class Test(unittest.TestCase):
    
    def close_enough(self, x, y):
        return (round(x * 1000) / 1000 == round(y * 1000) / 1000)

    def test_closs1(self):
        pred1 = [0.1, 0.2, 0.3, 0.3, 0.1] # 0.2, 0.1
        pred2 = [0.25, 0.1, 0.3, 0.05, 0.2] # 0.3 0.2
        pred3 = [0.05, 0.02, 0.02, 0.01, 0.9] # 0.01, 0.9
        gold = torch.tensor([1, 2, 3])
        preds = torch.tensor([pred1, pred2, pred3])
        criterion = ConfidenceLoss1(p0 = 0.5)
        criterion.notify(2)
        expected_loss = 1.0264
        assert self.close_enough(criterion(preds, gold).item(),
                                 expected_loss)


    def test_pairwise_confidence_loss(self):
        criterion = PairwiseConfidenceLoss('max_non_abs')
        #test compute_loss
        gold_probs_x = torch.tensor([0.2, 0.5, 1])   # probability of correct class from first network
        gold_probs_y = torch.tensor([0.5, 0.5, 0.2]) # probability of correct class from second network
        confidence_x = torch.tensor([0.5, 0.5, 0.8]) # confidences from first network
        confidence_y = torch.tensor([0.6, 0.6, 0.6]) # confidences from second network
        
        # softmax of (0.5, 0.6) == (.475, .525)
        # then expected loss for first component is:
        #   .475 * -log(0.2) + .525 * -log(.5)
        expected_losses = torch.tensor([1.1284, 0.6931, 0.7246])
        losses = criterion.compute_loss(confidence_x, confidence_y, 
                                        gold_probs_x, gold_probs_y)
        assert(torch.allclose(expected_losses, losses, atol=10**(-4)))
        

        #baseline
        output_x = torch.tensor([[1, 1, 0.],   # distribution 1  over classA, classB, abstain
                                 [1., 1., 1],     # distribution 2  over classA, classB, abstain
                                 [1, 0., 0.]])      # distribution 3  over classA, classB, abstain
        gold_x = torch.tensor([0, 1, 0])            # gold labels for instances 1, 2, 3
        output_y = torch.tensor([[1, 1, 0.],
                                 [1., 1., 1.],
                                 [1., 0, 0]])
        gold_y = torch.tensor([0, 0, 1])        
        expected_loss = torch.tensor(1.0040)
        loss = criterion(output_x, output_y, gold_x, gold_y)
        assert(torch.allclose(loss, expected_loss, atol=10**(-4)))

        #test loss function
        #inv_abs
        criterion = PairwiseConfidenceLoss('inv_abs')
       
        output_x = torch.tensor([[0.2, 0.2, 0.5],
                                 [0.5, 0.4, 0.5],
                                 [1, 0.2, 0.2]])
        output_y = torch.tensor([[0.5, 0.1, 0.4],
                                 [0.5, 0.1, 0.4],
                                 [0.2, 0.2, 0.4]])
        gold_x = [1, 0, 0]
        gold_y = [0, 0, 0]

        expected_loss = torch.tensor(0.9890)
        loss = criterion(output_x, output_y, gold_x, gold_y)
        assert(torch.allclose(loss, expected_loss, atol=10**(-4)))


if __name__ == '__main__':
        unittest.main()
