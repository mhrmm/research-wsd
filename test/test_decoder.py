import unittest
import torch
from reed_wsd.mnist.model import BasicFFN, AbstainingFFN
from reed_wsd.mnist.loader import MnistLoader
from reed_wsd.decoder import Decoder, AbstainingDecoder
from test.mnist.test_mnist_loader import load_mnist_data


class TestMnistDecoder(unittest.TestCase):

    def test_basic_decoder(self):
        torch.manual_seed(1977)
        net = BasicFFN()
        net.eval()
        decoder = Decoder()
        trainset = load_mnist_data()
        loader = MnistLoader(trainset, bsz=2, shuffle=False)
        decoded = decoder(net, loader)
        expected = [{'pred': 6, 'gold': 5, 'confidence': 0.1132},
                    {'pred': 7, 'gold': 0, 'confidence': 0.1128},
                    {'pred': 7, 'gold': 4, 'confidence': 0.1147},
                    {'pred': 8, 'gold': 1, 'confidence': 0.1125},
                    {'pred': 7, 'gold': 9, 'confidence': 0.1109}]
        result = []
        for i, x in enumerate(decoded):
            x['confidence'] = round(x['confidence'], 4)
            result.append(x)
            if i > 3:
                break
        assert result == expected

    def test_abstaining_decoder(self):
        torch.manual_seed(1977)
        net = AbstainingFFN(confidence_extractor='inv_abs')
        net.eval()
        decoder = AbstainingDecoder()
        trainset = load_mnist_data()
        loader = MnistLoader(trainset, bsz=2, shuffle=False)
        decoded = decoder(net, loader)
        expected = [{'pred': -1, 'gold': 5, 'confidence': 0.8992},
                    {'pred': -1, 'gold': 0, 'confidence': 0.8977},
                    {'pred': -1, 'gold': 4, 'confidence': 0.8982},
                    {'pred': -1, 'gold': 1, 'confidence': 0.9006},
                    {'pred': 1, 'gold': 9, 'confidence': 0.9019}]
        result = []
        for i, x in enumerate(decoded):
            x['confidence'] = round(x['confidence'], 4)
            result.append(x)
            if i > 3:
                break
        assert result == expected


if __name__ == "__main__":
    unittest.main()
