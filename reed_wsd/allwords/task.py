import os
from os.path import join
import torch.optim as optim
from torchvision import transforms
from reed_wsd.task import TaskFactory
from reed_wsd.allwords.model import SingleLayerFFNWithZones
from reed_wsd.allwords.model import AbstainingSingleLayerFFNWithZones, BEMforWSD
from reed_wsd.allwords.wordsense import SenseInstanceDataset, SenseTaggedSentences
from reed_wsd.allwords.wordsense import SenseInstanceLoader, TwinSenseInstanceLoader
from reed_wsd.allwords.vectorize import DiskBasedVectorManager
from reed_wsd.allwords.blevins import BEMDataset, BEMLoader
from reed_wsd.allwords.evaluate import AllwordsSimpleEmbeddingDecoder
from reed_wsd.allwords.evaluate import AllwordsAbstainingEmbeddingDecoder
from reed_wsd.allwords.evaluate import AllwordsBEMDecoder
from reed_wsd.allwords.train import SingleEmbeddingTrainer
from reed_wsd.allwords.train import PairwiseEmbeddingTrainer, BEMTrainer


file_dir = os.path.dirname(os.path.realpath(__file__))
mnist_dir = join(file_dir, 'mnist')
allwords_dir = join(file_dir, 'allwords')
allwords_data_dir = join(allwords_dir, 'data')
mnist_data_dir = join(mnist_dir, 'data')
mnist_train_dir = join(mnist_data_dir, 'train')
mnist_test_dir = join(mnist_data_dir, 'test')
imdb_dir = join(file_dir, 'imdb')
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                               ])


corpus_id_lookup = {'semcor': 'data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml',
                        'semev07': 'data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml',
                        'semev13': 'data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2013/semeval2013.data.xml',
                        'semev15': 'data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2015/semeval2015.data.xml',
                        'sensev2': 'data/WSD_Evaluation_Framework/Evaluation_Datasets/senseval2/senseval2.data.xml',
                        'sensev3': 'data/WSD_Evaluation_Framework/Evaluation_Datasets/senseval3/senseval3.data.xml'}


class AllwordsTaskFactory(TaskFactory):

    def __init__(self, config):
        super().__init__(config)

        self._model_lookup = {'simple': SingleLayerFFNWithZones,
                              'abstaining': AbstainingSingleLayerFFNWithZones,
                              'bem': BEMforWSD}
        self._decoder_lookup = {'simple': AllwordsSimpleEmbeddingDecoder,
                                'abstaining': AllwordsAbstainingEmbeddingDecoder,
                                'bem': AllwordsBEMDecoder}
        assert (self.config['task'] == 'allwords')

    @staticmethod
    def init_loader(stage, architecture, style, corpus_id, bsz):
        data_dir = allwords_data_dir
        filename = join(data_dir, 'raganato.json')
        sents = SenseTaggedSentences.from_json(filename, corpus_id)
        if architecture == "bem":
            ds = BEMDataset(sents)
            loader = BEMLoader(ds, bsz)
        if architecture == 'simple' or architecture == 'abstaining':
            vecmgr = DiskBasedVectorManager(join(join(data_dir, 'vecs'), corpus_id))
            ds = SenseInstanceDataset(sents, vecmgr)
            if stage == 'train':
                if style == 'single':
                    loader = SenseInstanceLoader(ds, batch_size=bsz)
                if style == 'pairwise':
                    loader = TwinSenseInstanceLoader(ds, batch_size=bsz)
            if stage == 'test':
                loader = SenseInstanceLoader(ds, batch_size=bsz)
        return loader

    def train_loader_factory(self):
        if self.config['architecture'] == 'bem' or self.config['architecture'] == 'simple':
            assert (self.config['style'] == 'single')
        return AllwordsTaskFactory.init_loader('train',
                                               self.config['architecture'],
                                               self.config['style'],
                                               corpus_id_lookup['semcor'])

    def val_loader_factory(self):
        if self.config['architecture'] == 'bem' or self.config['architecture'] == 'simple':
            assert (self.config['style'] == 'single')
        return AllwordsTaskFactory.init_loader('test',
                                               self.config['architecture'],
                                               self.config['style'],
                                               corpus_id_lookup[self.config['dev_corpus']])

    def decoder_factory(self):
        return self._decoder_lookup[self.config['architecture']]()

    def model_factory(self, data):
        if self.config['architecture'] == 'bem':
            model = self._model_lookup[self.config['architecture']](gpu=True)
        else:
            model = self._model_lookup[self.config['architecture']](input_size=768,
                                                                    output_size=data.num_senses(),
                                                                    zone_applicant=self.config['confidence'])
        return model

    def optimizer_factory(self, model):
        return optim.Adam(model.parameters(), lr=0.001)

    def select_trainer(self):
        if self.config['architecture'] == 'bem':
            trainer = BEMTrainer
        else:
            if self.config['style'] == 'single':
                trainer = SingleEmbeddingTrainer
            if self.config['style'] == 'pairwise':
                trainer = PairwiseEmbeddingTrainer
        return trainer

