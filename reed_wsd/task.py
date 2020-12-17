import torch
import torch.optim as optim
from reed_wsd.loss import CrossEntropyLoss, NLLLoss, AbstainingLoss
from reed_wsd.loss import ConfidenceLoss4, PairwiseConfidenceLoss, DACLoss
from reed_wsd.trustscore import TrustScore

criterion_lookup = {'crossentropy': CrossEntropyLoss,
                    'nll': NLLLoss,
                    'conf1': AbstainingLoss,
                    'conf4': ConfidenceLoss4,
                    'pairwise': PairwiseConfidenceLoss,
                    'dac': DACLoss}


class TaskFactory:
    def __init__(self, config):
        # check dependency
        assert (config['task'] in ['mnist', 'allwords', 'imdb'])
        if config['task'] in ['mnist', 'imdb']:
            assert (config['architecture'] != 'bem')
        if config['trustscore']:
            assert (config['confidence'] == 'max_prob')
        if config['architecture'] == 'simple':
            assert (config['confidence'] == 'max_prob')
            assert (config['criterion']['name'] == 'nll')
            assert (config['style'] == 'single')
        if config['architecture'] == 'abstaining':
            assert (config['confidence'] in ['max_non_abs', 'inv_abs', 'abs'])
            assert (config['criterion']['name'] in ['conf1', 'conf4', 'pairwise', 'dac'])
        if config['architecture'] == 'bem':
            assert (config['task'] == 'allwords')
            assert (config['criterion']['name'] == 'crossentropy')
            assert (config['style'] == 'single')
        if config['style'] == 'single':
            assert (config['criterion']['name'] in ['crossentropy', 'nll', 'conf1', 'conf4', 'dac'])
        if config['style'] == 'pairwise':
            assert (config['criterion']['name'] == 'pairwise')
            assert (config['architecture'] == 'abstaining')
        self.config = config

    def train_loader_factory(self):
        raise NotImplementedError("Cannot call on abstract class.")

    def val_loader_factory(self):
        raise NotImplementedError("Cannot call on abstract class.")

    def decoder_factory(self):
        raise NotImplementedError("Cannot call on abstract class.")

    def model_factory(self, data):
        raise NotImplementedError("Cannot call on abstract class.")

    def optimizer_factory(self):
        raise NotImplementedError("Cannot call on abstract class.")

    def criterion_factory(self):
        config = self.config['criterion']
        if config['name'] in ['conf1', 'conf4']:
            criterion = criterion_lookup[config['name']](alpha=config['alpha'], warmup_epochs=config['warmup_epochs'])
        if config['name'] in ['crossentropy', 'nll', 'pairwise']:
            criterion = criterion_lookup[config['name']]()
        if config['name'] in ['dac']:
            criterion = criterion_lookup[config['name']](target_alpha=config['alpha'],
                                                         warmup_epochs=config['warmup_epochs'],
                                                         total_epochs=self.config['n_epochs'],
                                                         alpha_init_factor=config['alpha_init_factor'])
        return criterion

    def trust_model_factory(self, train_loader):
        if self.config['trustscore']:
            trust_model = TrustScore()
            train_instances = list(train_loader)
            train_evidence = torch.cat([evidence for evidence, label in train_instances]).numpy()
            train_label = torch.cat([label for evidence, label in train_instances]).numpy()
            trust_model.fit(train_evidence, train_label)
            return trust_model
        else:
            return None

    def scheduler_factory(self, optimizer):
        if self.config['criterion']['name'] == 'dac':
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80, 120], gamma=0.5)
        else:
            return None

    def select_trainer(self):
        raise NotImplementedError("Cannot call on abstract class.")

    def trainer_factory(self):
        train_loader = self.train_loader_factory()
        if self.config['trustscore']:
            trustmodel = self.trust_model_factory(self, train_loader)
        else:
            trustmodel = None
        val_loader = self.val_loader_factory()
        decoder = self.decoder_factory()
        model = self.model_factory(train_loader)
        optimizer = self.optimizer_factory(model)
        scheduler = self.scheduler_factory(optimizer)
        criterion = self.criterion_factory()
        n_epochs = self.config['n_epochs']
        trainer_class = self.select_trainer()
        trainer = trainer_class(criterion, optimizer, train_loader,
                                val_loader, decoder, n_epochs, trustmodel, scheduler)

        print('model:', type(model).__name__)
        print('criterion:', type(criterion).__name__)
        print('optimizer:', type(optimizer).__name__)
        print('train loader:', type(train_loader).__name__)
        print('val loader:', type(val_loader).__name__)
        print('decoder:', type(decoder).__name__)
        print('n_epochs', n_epochs)
        if trustmodel is not None:
            print('trustscore:', True)
        else:
            print('trustscore:', False)
        if hasattr(criterion, 'warmup_epochs'):
            print('warmup epochs:', criterion.warmup_epochs)
        else:
            print('warmup epochs: N/A')

        return trainer, model

