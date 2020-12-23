import json
import copy
import sys
from reed_wsd.allwords.task import AllwordsTaskFactory
from reed_wsd.mnist.task import MnistTaskFactory
from reed_wsd.imdb.task import IMDBTaskFactory

task_factories = {'mnist': MnistTaskFactory,
                  'allwords': AllwordsTaskFactory,
                  'imdb': IMDBTaskFactory}


class Experiment:

    def __init__(self, config, reps=1):
        self.config = config
        self.reps = reps
        self.task_factory = task_factories[config['task']['name']](config)
        self.result = None

    def run(self):
        all_analytics = []
        for i in range(self.reps):
            print('\nTRIAL {}'.format(i))
            trainer, model = self.task_factory.trainer_factory()    
            _, analytics = trainer(model)
            analytics.show_training_dashboard()
            all_analytics.append(analytics)
        # self.result = Analytics.average_list_of_results(all_analytics)

    def return_analytics(self):
        return copy.deepcopy(self.result)


class ExperimentSequence:
    def __init__(self, experiments, reps):
        self.experiments = experiments
        self.reps = reps
    
    @classmethod
    def from_json(cls, configs_path, reps=1):
        with open(configs_path, 'r') as f:
            configs = json.load(f)
        experiments = []
        for config in configs:
            experiments.append(Experiment(config, reps))
        return cls(experiments, reps)

    def run_and_save(self, out_path):
        results = []
        for experiment in self.experiments:
            experiment.run()
            results.append((experiment.config,
                            experiment.return_analytics()))
            with open(out_path, 'w') as f:
                json.dump(results, f)


if __name__ == "__main__":
    config_path = sys.argv[1]
    output_path = sys.argv[2]

    exp_seq = ExperimentSequence.from_json(config_path)
    exp_seq.run_and_save(output_path)
