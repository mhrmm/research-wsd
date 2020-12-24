import json
from reed_wsd.allwords.task import AllwordsTaskFactory
from reed_wsd.mnist.task import MnistTaskFactory
from reed_wsd.imdb.task import IMDBTaskFactory
from analytics import ResultDatabase

task_factories = {'mnist': MnistTaskFactory,
                  'allwords': AllwordsTaskFactory,
                  'imdb': IMDBTaskFactory}


class Experiment:

    def __init__(self, config):
        self.config = config
        self.task_factory = task_factories[config['task']['name']](config)

    def run(self):
        trainer, model = self.task_factory.trainer_factory()
        _, result = trainer(model)
        result.show_training_dashboard()
        return result


class ExperimentSequence:
    def __init__(self, experiments):
        self.experiments = experiments
    
    @classmethod
    def from_json(cls, configs_path):
        with open(configs_path, 'r') as f:
            configs = json.load(f)
        experiments = []
        for config in configs:
            experiments.append(Experiment(config))
        return cls(experiments)

    def run(self):
        results = []
        for experiment in self.experiments:
            result = experiment.run()
            results.append(result)
        return ResultDatabase(results)
