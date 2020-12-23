import torch
from torch.nn import functional
from reed_wsd.util import cudaify, ABS
from reed_wsd.train import Decoder
from tqdm import tqdm


class MnistSimpleDecoder(Decoder):

    def get_loss(self):
        return self.running_loss_total / self.running_loss_denom

    def __call__(self, net, data, loss_f=None, trust_model=None):  # TODO: put trust_model into confidence
        net.eval()
        self.running_loss_total = 0.0
        self.running_loss_denom = 0
        for images, labels in tqdm(data, total=len(data)):
            with torch.no_grad():
                outputs, conf = net(cudaify(images))
            if loss_f is not None:
                loss = loss_f(outputs, conf, cudaify(labels))
                self.running_loss_total += loss.item()
                self.running_loss_denom += 1  # TODO: why 1 and not len(images)?
            preds = outputs.argmax(dim=1)
            if trust_model is not None:
                trust_score = trust_model.get_score(images.cpu().numpy(),
                                                    preds.cpu().numpy())
            else:
                trust_score = [None] * labels.shape[0]
            for element in zip(preds, labels, conf, trust_score):
                p, g, c, t = element
                if t is not None:
                    yield {'pred': p.item(), 'gold': g.item(), 'confidence': c.item(), 'trustscore': t}
                else:
                    yield {'pred': p.item(), 'gold': g.item(), 'confidence': c.item()}


class MnistAbstainingDecoder(Decoder):  # TODO: is this needed (or even correct)?

    def get_loss(self):
        if self.running_loss_denom == 0:
            return None
        return self.running_loss_total / self.running_loss_denom

    def __call__(self, net, data, loss_f=None, trust_model=None):
        net.eval()
        self.running_loss_total = 0.0
        self.running_loss_denom = 0
        for images, labels in tqdm(data, total=len(data)):
            with torch.no_grad():
                output, conf = net(cudaify(images))
            if loss_f is not None:
                loss = loss_f(output, conf, cudaify(labels))
                self.running_loss_total += loss.item()
                self.running_loss_denom += 1
            output = functional.softmax(output.clamp(min=-25, max=25), dim=1)
            abs_i = output.shape[1] - 1
            preds = output.argmax(dim=-1)
            preds[preds == abs_i] = ABS
            for e in zip(preds, labels, conf):
                pred, gold, c = e
                result = {'pred': pred.item(), 'gold': gold.item(),
                          'confidence': c.item()}
                yield result
