import unittest
import torch
from reed_wsd.mnist.model import BasicFFN, AbstainingFFN, ConfidentFFN
from reed_wsd.mnist.model import inv_abstain_prob, max_nonabstain_prob


def set_ffn_params(net):
    for param in net.parameters():
        if param.shape == torch.Size([3]):
            param[0] = 1.4640
            param[1] = -0.3238
            param[2] = 0.7740
        elif param.shape == torch.Size([2, 2]):
            param[0][0] = 0.1940
            param[0][1] = 2.1614
            param[1][0] = -0.1721
            param[1][1] = -0.1721
        elif param.shape == torch.Size([2]):
            param[0] = 0.1391
            param[1] = -0.1082
        elif param.shape == torch.Size([3, 2]):
            param[0][0] = -1.2682
            param[0][1] = -0.0383
            param[1][0] = -0.1029
            param[1][1] = 1.4400
            param[2][0] = -0.4705
            param[2][1] = 1.1624
        else:
            torch.nn.init.ones_(param)


def build_basic_ffn():
    net = BasicFFN(input_size=2, hidden_sizes=[2, 2], output_size=2)
    set_ffn_params(net)
    net.eval()
    return net


def run_basic_input(net):
    result, conf = net(torch.tensor([[-2., 1.], [5., 2.]]))
    result = torch.softmax(result.to('cpu'), dim=1)
    expected = torch.tensor([[0.6188, 0.3812],
                             [0.6646, 0.3354]]).to('cpu')
    assert result.shape == expected.shape
    assert torch.allclose(result, expected, atol=10 ** (-4))
    return conf


def build_abstaining_ffn(confidence):
    net = AbstainingFFN(input_size=2, hidden_sizes=[2, 2], output_size=2)
    set_ffn_params(net)
    if confidence is not None:
        net.confidence_extractor = confidence
    net.eval()
    return net


def run_abstaining_input(net):
    result, conf = net(torch.tensor([[-2., 1.], [5., 2.]]))
    result = torch.softmax(result.to('cpu'), dim=1)
    expected = torch.tensor([[0.4551, 0.1621, 0.3828],
                             [0.3366, 0.2261, 0.4372]]).to('cpu')
    assert result.shape == expected.shape
    assert torch.allclose(result, expected, atol=10 ** (-4))
    return conf


def build_confident_ffn():
    net = ConfidentFFN(input_size=2, hidden_sizes=[2, 2], output_size=2)
    set_ffn_params(net)
    net.eval()
    return net


class TestMnistNetworks(unittest.TestCase):

    def test_basic_ffn(self):
        net = build_basic_ffn()
        conf = run_basic_input(net)
        expected_conf = torch.tensor([0.6188, 0.6646]).to('cpu')
        assert conf.shape == expected_conf.shape
        assert torch.allclose(conf, expected_conf, atol=10 ** (-4))

    def test_abstaining_ffn(self):
        net = build_abstaining_ffn(confidence=None)
        run_abstaining_input(net)
    
    def test_abstaining_ffn_with_inv_abstain_prob(self):
        net = build_abstaining_ffn(confidence=inv_abstain_prob)
        conf = run_abstaining_input(net).to('cpu')
        expected_conf = torch.tensor([0.6172, 0.5628]).to('cpu')
        assert conf.shape == expected_conf.shape
        assert torch.allclose(conf, expected_conf, atol=10**(-4))
        
    def test_abstaining_ffn_with_max_nonabstain_prob(self):
        net = build_abstaining_ffn(confidence=max_nonabstain_prob)
        conf = run_abstaining_input(net).to('cpu')
        expected_conf = torch.tensor([0.4551, 0.3366]).to('cpu')
        assert conf.shape == expected_conf.shape
        assert torch.allclose(conf, expected_conf, atol=10**(-4))

    def test_confident_ffn(self):
        net = build_confident_ffn()
        conf = run_basic_input(net)
        expected_conf = torch.tensor([1.6482, 2.1929]).to('cpu')
        assert conf.shape == expected_conf.shape
        assert torch.allclose(conf, expected_conf, atol=10**(-4))


if __name__ == "__main__":
    unittest.main()
