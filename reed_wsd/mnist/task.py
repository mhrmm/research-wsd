import os
from os.path import join
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from reed_wsd.task import TaskFactory
from reed_wsd.mnist.model import BasicFFN, AbstainingFFN
from reed_wsd.mnist.decoder import MnistSimpleDecoder, MnistAbstainingDecoder
from reed_wsd.mnist.train import SingleTrainer as MnistSingleTrainer
from reed_wsd.mnist.train import PairwiseTrainer as MnistPairwiseTrainer
from reed_wsd.mnist.loader import MnistLoader, ConfusedMnistLoader
from reed_wsd.mnist.loader import MnistPairLoader, ConfusedMnistPairLoader


mnist_dir = os.path.dirname(os.path.realpath(__file__))
mnist_data_dir = join(mnist_dir, 'data')
mnist_train_dir = join(mnist_data_dir, 'train')
mnist_test_dir = join(mnist_data_dir, 'test')
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                               ])


class MnistTaskFactory(TaskFactory):

    def __init__(self, config):
        super().__init__(config)
        self._model_lookup = {'simple': BasicFFN,
                              'abstaining': AbstainingFFN}
        self._decoder_lookup = {'simple': MnistSimpleDecoder,
                                'abstaining': MnistAbstainingDecoder}

    @staticmethod
    def init_loader(stage, style, confuse, bsz):
        if stage == 'train':
            ds = datasets.MNIST(mnist_train_dir, download=True, train=True, transform=transform)
            if style == 'single':
                if confuse != False:
                    loader = ConfusedMnistLoader(ds, bsz, confuse, shuffle=True)
                else:
                    loader = MnistLoader(ds, bsz, shuffle=True)
            if style == 'pairwise':
                if confuse != False:
                    loader = ConfusedMnistPairLoader(ds, bsz, confuse, shuffle=True)
                else:
                    loader = MnistPairLoader(ds, bsz, shuffle=True)
        if stage == 'test':
            ds = datasets.MNIST(mnist_test_dir, download=True, train=False, transform=transform)
            if confuse != False:
                loader = ConfusedMnistLoader(ds, bsz, confuse, shuffle=True)
            else:
                loader = MnistLoader(ds, bsz, shuffle=True)
        return loader

    def train_loader_factory(self):
        return MnistTaskFactory.init_loader('train',
                                            self.config['style'],
                                            self.config['confuse'],
                                            self.config['bsz'])

    def val_loader_factory(self):
        return MnistTaskFactory.init_loader('test',
                                            self.config['style'],
                                            self.config['confuse'],
                                            self.config['bsz'])

    def decoder_factory(self):
        return self._decoder_lookup[self.config['architecture']]()

    def model_factory(self, data):
        return self._model_lookup[self.config['architecture']](confidence_extractor=self.config['confidence'])

    def optimizer_factory(self, model):
        if self.config['criterion']['name'] == 'dac':
            return optim.SGD(model.parameters(), lr=0.02, weight_decay=5e-4, nesterov=True, momentum=0.9)
        else:
            return optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    def select_trainer(self):
        if self.config['style'] == 'single':
            trainer = MnistSingleTrainer
        elif self.config['style'] == 'pairwise':
            trainer = MnistPairwiseTrainer
        return trainer

