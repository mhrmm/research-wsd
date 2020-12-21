import unittest
from math import log
import torch
from torch import tensor
from reed_wsd.loss import NLLLoss, PairwiseConfidenceLoss, CrossEntropyLoss
from reed_wsd.loss import AbstainingLoss
import torch.nn.functional as F


def approx(x, y, num_digits=4):
    return abs(x-y) < 1.0 * (10 ** -num_digits)


def softmax(t):
    return F.softmax(t.clamp(min=-25, max=25), dim=1)


def close_enough(t1, t2):
    error_msg = "Not close enough:\n{}\n{}".format(t1, t2)
    assert torch.allclose(t1, t2, atol=0.001), error_msg


class TestMnistLoss(unittest.TestCase):
    def test_nll_loss1(self):
        predictions = tensor([[-1., -2., -3.]])
        loss_function = NLLLoss()
        gold = torch.tensor([2])
        loss = loss_function(predictions, None, gold)
        close_enough(loss, tensor(3.))

    def test_nll_loss2(self):
        predictions = tensor([[-1., -2., -3.],
                              [4., 5., 6.]])
        loss_function = NLLLoss()
        gold = torch.tensor([2, 1])
        loss = loss_function(predictions, None, gold)
        close_enough(loss, tensor((3-5)/2.))

    def test_cross_entropy_loss1(self):
        predictions = tensor([[-1., -2., -3.]])
        predicted_probs = softmax(predictions)
        close_enough(predicted_probs, tensor([[0.6652, 0.2447, 0.0900]]))
        loss_function = CrossEntropyLoss()
        gold = torch.tensor([2])
        loss = loss_function(predictions, None, gold)
        close_enough(loss, tensor(-log(.0900)))

    def test_cross_entropy_loss2(self):
        predictions = tensor([[-1., -2., -3.],
                              [-1., -2., -3.]])
        predicted_probs = softmax(predictions)
        close_enough(predicted_probs, tensor([[0.6652, 0.2447, 0.0900],
                                              [0.6652, 0.2447, 0.0900]]))
        loss_function = CrossEntropyLoss()
        gold = torch.tensor([2, 1])
        loss = loss_function(predictions, None, gold)
        close_enough(loss, tensor((-log(.0900) - log(.2447))/2))

    def test_abstaining_loss1(self):
        in_vec = torch.tensor([[0.25, 0.25, 0.1, 0.4]])
        gold = torch.tensor([2])
        loss_function = AbstainingLoss(alpha=0.5)
        loss_function.notify(5)
        loss = loss_function(in_vec, None, gold)
        print(loss)
        expected_loss = torch.tensor(1.2039728043259361)  # i.e., -log(0.3)
        assert (torch.allclose(loss, expected_loss, atol=0.0001))

    def test_abstaining_loss2(self):
        in_vec = torch.tensor([[0.3, 0.2, 0.1, 0.4],
                               [0.1, 0.1, 0.6, 0.2]])
        gold = torch.tensor([2, 1])
        loss_function = AbstainingLoss(alpha=0.5)
        loss_function.notify(5)
        loss = loss_function(in_vec, None, gold)

"""
class TestPairwiseConfidenceLoss(unittest.TestCase):
    def test_call(self):
        criterion = PairwiseConfidenceLoss()
        in_vec_x = torch.tensor([[0.3, 0.2, 0.5],
                                 [0.1, 0.1, 0.8]])
        gold_x = torch.tensor([1, 1])
        in_vec_y = torch.tensor([[0.6, 0.3, 0.1],
                                 [0.3, 0.3, 0.4]])
        gold_y = torch.tensor([1, 1])
        conf_x = torch.tensor([0.5, 0.2])
        conf_y = torch.tensor([0.9, 0.6])

        expected_loss_x = 1.3667
        expected_loss_y = 1.6448

        expected_loss = torch.tensor((expected_loss_x + expected_loss_y) / 2)
        loss = criterion(in_vec_x, in_vec_y, gold_x, gold_y, conf_x, conf_y)
        assert (torch.allclose(expected_loss, loss, atol=0.0001))


class TestLoss(unittest.TestCase):
    def test_nll_loss(self):
        loss = torch.nn.NLLLoss()
        predicted = tensor([[1, 0, 0, 0.], [0, 1, 0, 0.]])
        gold = tensor([0, 1])
        assert(loss(predicted, gold).item() == -1.0)
        predicted = tensor([[0.8, 0, 0, 0.2], [0.2, 0.8, 0, 0.]])
        gold = tensor([0, 1])
        assert(approx(loss(predicted, gold).item(), -0.8))
        predicted = tensor([[0.2, 0, 0, 0.8], [0.2, 0.8, 0, 0.]])
        gold = tensor([0, 1])
        assert(approx(loss(predicted, gold).item(), -0.5))

    def test_pairwise_confidence_loss(self):
        criterion = PairwiseConfidenceLoss()
        output_x = torch.tensor([[0.2, 0.5, 0.3],   # distribution 1  over classA, classB, abstain
                                 [0.5, 0.5, 0]])    # distribution 2  over classA, classB, abstain
        gold_x = torch.tensor([0, 1])            # gold labels for instances 1, 2
        output_y = torch.tensor([[0.5, 0.2, 0.3],
                                 [0.5, 0.1, 0.4]])
        gold_y = torch.tensor([1, 0])        
        confidence_x = torch.tensor([0., 0.]) # confidences from first network
        confidence_y = torch.tensor([0., 0.]) # confidences from second network

        # loss should be -log [ ((0.2 * 0.5 + 0.2 * 0.5) + (0.5 * 0.5 + 0.5 * 0.5)) / 2 ]
        # which equals 1.0498
        expected_loss = torch.tensor(1.0498)
        loss = criterion(output_x, output_y, gold_x, gold_y, confidence_x, confidence_y)
        assert(torch.allclose(loss, expected_loss, atol=10**(-4)))

    def test_pairwise_confidence_loss2(self):
        criterion = PairwiseConfidenceLoss()
        output_x = torch.tensor([[0.2, 0.5, 0.3],   # distribution 1  over classA, classB, abstain
                                 [0.5, 0.5, 0]])    # distribution 2  over classA, classB, abstain
        gold_x = torch.tensor([1, 0])            # gold labels for instances 1, 2
        output_y = torch.tensor([[0.5, 0.2, 0.3],
                                 [0.5, 0.1, 0.4]])
        gold_y = torch.tensor([1, 0])        
        confidence_x = torch.tensor([0., 0.]) # confidences from first network
        confidence_y = torch.tensor([0., 0.]) # confidences from second network

        # loss should be -log [ ((0.5 * 0.5 + 0.2 * 0.5) + (0.5 * 0.5 + 0.5 * 0.5)) / 2 ]
        # which equals 0.8557
        expected_loss = torch.tensor(0.8557)
        loss = criterion(output_x, output_y, gold_x, gold_y, confidence_x, confidence_y)
        assert(torch.allclose(loss, expected_loss, atol=10**(-4)))

    def test_pairwise_confidence_loss3(self):
        criterion = PairwiseConfidenceLoss()
        output_x = torch.tensor([[0.2, 0.3, 0.5],   # distribution 1  over classA, classB, abstain
                                 [0.1, 0.5, 0.4]])    # distribution 2  over classA, classB, abstain
        gold_x = torch.tensor([1, 0]) # assigned gold probs are 0.3 and 0.1
        output_y = torch.tensor([[0.4, 0.5, 0.1],
                                 [0.6, 0.2, 0.2]])
        gold_y = torch.tensor([0, 1]) # assigned gold probs are 0.4 and 0.2
        confidence_x = torch.tensor([-math.log(2), math.log(3)]) # confidences from first network
        confidence_y = torch.tensor([math.log(2), -math.log(3)]) # confidences from second network

        # the weighted prob of instance 1 is 0.2 * 0.3 + 0.8 * 0.4
        # the weighted prob of instance 2 is 0.9 * 0.4 + 0.1 * 0.2
        # loss should be -log [ ((0.2 * 0.3 + 0.8 * 0.4) + (0.9 * 0.4 + 0.1 * 0.2)) / 2 ]
        # which equals 0.8557
        expected_loss = torch.tensor(1.406497)
        loss = criterion(output_x, output_y, gold_x, gold_y, confidence_x, confidence_y)
        assert(torch.allclose(loss, expected_loss, atol=10**(-4)))


class TestAnotherLoss(unittest.TestCase):

    def close_enough(self, x, y):
        return (round(x * 1000) / 1000 == round(y * 1000) / 1000)

    def test_closs1(self):
        pred1 = [0.1, 0.2, 0.3, 0.3, 0.1]  # 0.2, 0.1
        pred2 = [0.25, 0.1, 0.3, 0.05, 0.2]  # 0.3 0.2
        pred3 = [0.05, 0.02, 0.02, 0.01, 0.9]  # 0.01, 0.9
        gold = torch.tensor([1, 2, 3])
        preds = torch.tensor([pred1, pred2, pred3])
        criterion = AbstainingLoss(p0=0.5)
        criterion.notify(2)
        expected_loss = 1.0264
        assert self.close_enough(criterion(preds, gold).item(),
                                 expected_loss)

    def test_pairwise_confidence_loss(self):
        criterion = PairwiseConfidenceLoss('max_non_abs')
        # test compute_loss
        gold_probs_x = torch.tensor([0.2, 0.5, 1])  # probability of correct class from first network
        gold_probs_y = torch.tensor([0.5, 0.5, 0.2])  # probability of correct class from second network
        confidence_x = torch.tensor([0.5, 0.5, 0.8])  # confidences from first network
        confidence_y = torch.tensor([0.6, 0.6, 0.6])  # confidences from second network

        # softmax of (0.5, 0.6) == (.475, .525)
        # then expected loss for first component is:
        #   .475 * -log(0.2) + .525 * -log(.5)
        expected_losses = torch.tensor([1.1284, 0.6931, 0.7246])
        losses = criterion.compute_loss(confidence_x, confidence_y,
                                        gold_probs_x, gold_probs_y)
        assert (torch.allclose(expected_losses, losses, atol=10 ** (-4)))

        # baseline
        output_x = torch.tensor([[1, 1, 0.],  # distribution 1  over classA, classB, abstain
                                 [1., 1., 1],  # distribution 2  over classA, classB, abstain
                                 [1, 0., 0.]])  # distribution 3  over classA, classB, abstain
        gold_x = torch.tensor([0, 1, 0])  # gold labels for instances 1, 2, 3
        output_y = torch.tensor([[1, 1, 0.],
                                 [1., 1., 1.],
                                 [1., 0, 0]])
        gold_y = torch.tensor([0, 0, 1])
        expected_loss = torch.tensor(1.0040)
        loss = criterion(output_x, output_y, gold_x, gold_y)
        assert (torch.allclose(loss, expected_loss, atol=10 ** (-4)))

        # test loss function
        # inv_abs
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
        assert (torch.allclose(loss, expected_loss, atol=10 ** (-4)))
"""

if __name__ == "__main__":
    unittest.main()

