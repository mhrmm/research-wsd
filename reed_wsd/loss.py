import torch
import torch.nn.functional as F
from reed_wsd.util import abstract_method


class ConfidenceLoss(torch.nn.Module):
    def __init__(self):
        super(ConfidenceLoss, self).__init__()

    def notify(self, epoch):
        pass


class SingleConfidenceLoss(ConfidenceLoss):
    def __init__(self):
        super(SingleConfidenceLoss, self).__init__()

    def __call__(self, output, confidence, gold):
        abstract_method()


class CrossEntropyLoss(SingleConfidenceLoss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def __call__(self, output, confidence, gold):
        return self.loss(output, gold)

    def __str__(self):
        return "CrossEntropyLoss"


class NLLLoss(SingleConfidenceLoss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.NLLLoss()

    def __call__(self, output, confidence, gold):
        return self.loss(output, gold)

    def __str__(self):
        return "NLLLoss"


class AbstainingLoss(SingleConfidenceLoss):
    def __init__(self, alpha=0.5, warmup_epochs=3):
        super().__init__()
        self.alpha = 0.0
        self.warmup_epochs = warmup_epochs
        self.target_alpha = alpha
        self.notify(0)

    def notify(self, epoch):
        if epoch >= self.warmup_epochs:
            self.alpha = self.target_alpha

    def __call__(self, output, confidence, gold):
        output = F.softmax(output.clamp(min=-25, max=25), dim=1)
        label_ps = output[list(range(output.shape[0])), gold]
        abstains = output[:, -1]
        losses = label_ps + (self.alpha * abstains)
        losses = torch.clamp(losses, min=0.000000001)
        return -torch.mean(torch.log(losses))


class ConfidenceLoss4(SingleConfidenceLoss):
    def __init__(self, alpha=0.5, warmup_epochs=5):
        super().__init__()
        self.alpha = 0.0
        self.warmup_epochs = warmup_epochs
        self.target_alpha = alpha
        self.notify(0)

    def notify(self, epoch):
        if epoch >= self.warmup_epochs:
            self.alpha = self.target_alpha

    def __call__(self, output, confidence, gold):
        label_ps = output[list(range(len(output))), gold]
        label_ps_woa = output[:, :-1]
        label_ps_woa = F.normalize(label_ps_woa, p=1, dim=1)
        label_ps_woa = label_ps_woa[list(range(len(label_ps_woa))), gold]
        losses = label_ps_woa * (label_ps + (self.alpha * confidence))
        losses = torch.clamp(losses, min=0.000000001)
        return -torch.mean(torch.log(losses))

    def __str__(self):
        return "ConfidenceLoss4_p0_" + str(self.p0)


class PairwiseConfidenceLoss(ConfidenceLoss):

    def __call__(self, output_x, output_y, gold_x, gold_y, conf_x, conf_y):
        def confidence_weighted_loss(confidence_x, confidence_y, nll_x, nll_y):
            confidence_pair = torch.stack([confidence_x, confidence_y], dim=-1)
            softmaxed_pair = F.softmax(confidence_pair, dim=-1)
            nll_pair = torch.stack([nll_x, nll_y], dim=-1)
            return torch.sum(nll_pair * softmaxed_pair, dim=-1)

        loss = torch.nn.NLLLoss()
        output_x = F.softmax(output_x.clamp(min=-25, max=25), dim=1)
        output_y = F.softmax(output_y.clamp(min=-25, max=25), dim=1)
        nll_x = -loss(output_x, gold_x)
        nll_y = -loss(output_y, gold_y)
        losses = confidence_weighted_loss(conf_x, conf_y, nll_x, nll_y)
        return -torch.log(losses.mean())
