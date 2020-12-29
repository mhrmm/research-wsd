import os
from os.path import join
import torch.optim as optim
from torchvision import transforms
from reed_wsd.task import TaskFactory
from reed_wsd.decoder import InterfaceADecoder, InterfaceBDecoder
from reed_wsd.mnist.train import SingleTrainer as MnistSingleTrainer
from reed_wsd.mnist.train import PairwiseTrainer as MnistPairwiseTrainer
from reed_wsd.imdb.loader import IMDBDataset, IMDBLoader, IMDBTwinLoader
from reed_wsd.imdb.model import SingleLayerFFN, AbstainingSingleLayerFFN

file_dir = os.path.dirname(os.path.realpath(__file__))
imdb_dir = join(file_dir, 'imdb')
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                               ])


class IMDBTaskFactory(TaskFactory):
    def __init__(self, config):
        super().__init__(config)
        self._model_lookup = {'simple': SingleLayerFFN,
                              'abstaining': AbstainingSingleLayerFFN}
        self._decoder_lookup = {'simple': InterfaceADecoder,
                                'abstaining': InterfaceBDecoder}

    def train_loader_factory(self):
        ds = IMDBDataset.from_json(join(imdb_dir, 'data/aclImdb/imdb.json'), 'train')
        bsz = self.config['bsz']
        if self.config['style'] == 'single':
            loader = IMDBLoader(ds, bsz, shuffle=True)
        if self.config['style'] == 'pairwise':
            loader = IMDBTwinLoader(ds, bsz)
        return loader

    def val_loader_factory(self):
        ds = IMDBDataset.from_json(join(imdb_dir, 'data/aclImdb/imdb.json'), 'test')
        bsz = self.config['bsz']
        loader = IMDBLoader(ds, bsz, shuffle=True)
        return loader

    def decoder_factory(self):
        return self._decoder_lookup[self.config['architecture']]()

    def model_factory(self, data):
        return self._model_lookup[self.config['architecture']](confidence_extractor=self.config['confidence'])

    def optimizer_factory(self, model):
        if self.config['criterion']['name'] == 'dac':
            return optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, nesterov=True, momentum=0.9)
        else:
            return optim.Adam(model.parameters(), lr=0.0005)

    def scheduler_factory(self, optimizer):
        if self.config['criterion']['name'] == 'dac':
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 60, 120], gamma=0.1)
        else:
            return None

    def select_trainer(self):
        if self.config['style'] == 'single':
            trainer = MnistSingleTrainer
        elif self.config['style'] == 'pairwise':
            trainer = MnistPairwiseTrainer
        return trainer