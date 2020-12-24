import torch
import torch.optim as optim
from reed_wsd.loss import CrossEntropyLoss, NLLLoss, AbstainingLoss
from reed_wsd.loss import ConfidenceLoss4, PairwiseConfidenceLoss
from reed_wsd.dac import DACLoss
from reed_wsd.trustscore import TrustScore
from abc import ABC


class TaskFactory(ABC):
    def __init__(self, config):
        self.criterion_lookup = {'crossentropy': CrossEntropyLoss,
                                 'nll': NLLLoss,
                                 'conf1': AbstainingLoss,
                                 'conf4': ConfidenceLoss4,
                                 'pairwise': PairwiseConfidenceLoss,
                                 'dac': DACLoss}
        self.config = config
        # self.conduct_sanity_check()

    def conduct_sanity_check(self):
        assert (self.config['task']['name'] in ['mnist', 'allwords', 'imdb'])
        if self.config['task']['name'] in ['mnist', 'imdb']:
            assert (self.config['architecture'] != 'bem')
        if self.config['trustscore']:
            assert (self.config['confidence'] == 'max_prob')
        if self.config['architecture'] == 'simple':
            assert (self.config['confidence'] == 'max_prob')
            assert (self.config['criterion']['name'] == 'nll')
            assert (self.config['style'] == 'single')
        if self.config['architecture'] == 'abstaining':
            assert (self.config['confidence'] in ['max_non_abs', 'inv_abs', 'abs'])
            assert (self.config['criterion']['name'] in ['conf1', 'conf4', 'pairwise', 'dac'])
        if self.config['architecture'] == 'bem':
            assert (self.config['task'] == 'allwords')
            assert (self.config['criterion']['name'] == 'crossentropy')
            assert (self.config['style'] == 'single')

    def train_loader_factory(self):
        raise NotImplementedError("Cannot call on abstract class.")

    def val_loader_factory(self):
        raise NotImplementedError("Cannot call on abstract class.")

    def decoder_factory(self):
        raise NotImplementedError("Cannot call on abstract class.")

    def model_factory(self, data):
        raise NotImplementedError("Cannot call on abstract class.")

    def select_trainer(self):
        raise NotImplementedError("Cannot call on abstract class.")

    def trainer_factory(self):
        train_loader = self.train_loader_factory()
        trustmodel = self.trust_model_factory(train_loader)
        val_loader = self.val_loader_factory()
        decoder = self.decoder_factory()
        model = self.model_factory(train_loader)
        optimizer = self.optimizer_factory(model)
        scheduler = self.scheduler_factory(optimizer)
        loss = self.loss_factory()
        n_epochs = self.config['trainer']['n_epochs']
        trainer_class = self.select_trainer()
        trainer = trainer_class(self.config, loss, optimizer, train_loader,
                                val_loader, decoder, n_epochs, trustmodel, scheduler)
        return trainer, model

    def optimizer_factory(self, model):
        optim_constrs = {'sgd': optim.SGD}
        oconfig = self.config['trainer']['optimizer']
        optim_constr = optim_constrs[oconfig['name']]
        params = {k: v for k, v in oconfig.items() if k != 'name'}
        return optim_constr(model.parameters(), **params)

    def loss_factory(self):
        lconfig = self.config['trainer']['loss']
        params = {k: v for k, v in lconfig.items() if k != 'name'}
        return self.criterion_lookup[lconfig['name']](**params)

    def trust_model_factory(self, train_loader):
        if 'trustscore' in self.config and self.config['trustscore']:
            trust_model = TrustScore()
            train_instances = list(train_loader)
            train_evidence = torch.cat([evidence for evidence, label in train_instances]).numpy()
            train_label = torch.cat([label for evidence, label in train_instances]).numpy()
            trust_model.fit(train_evidence, train_label)
            return trust_model
        else:
            return None

    def scheduler_factory(self, optimizer):
        if self.config['trainer']['loss']['name'] == 'dac':
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80, 120], gamma=0.5)
        else:
            return None
