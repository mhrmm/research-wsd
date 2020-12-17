import json
import copy
import sys
from functools import reduce
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
        self.task_factory = task_factories[config['task']](config)
        self.result = None

    def run(self):
        measurements = []
        for i in range(self.reps):
            print('\nTRIAL {}'.format(i))
            trainer, model = self.task_factory.trainer_factory()    
            _, results = trainer(model)
            measurements.append(results)
        measurement_sum = reduce(Experiment.add_analytics_dict, measurements)
        avg_measurement = Experiment.div_analytics_dict(measurement_sum, self.reps)
        self.result = avg_measurement
        return results
    
    def return_analytics(self):
        result = copy.deepcopy(self.result)
        return result

    @staticmethod
    def add_analytics_dict(this, other):
        def elementwise_add(ls1, ls2):
            assert (len(ls1) == len(ls2))
            return [(ls1[i] + ls2[i]) for i in range(len(ls1))]

        def process_key(key):
            if key != 'prediction_by_class':
                return this[key] + other[key]
            else:
                return {key: elementwise_add(this[key], other[key]) for key in this.keys()}

        assert (this.keys() == other.keys())
        return {key: process_key(key) for key in this.keys()}

    @staticmethod
    def div_analytics_dict(d, divisor):
        def elementwise_div(ls):
            return [element / divisor for element in ls]

        def process_key(key):
            if key != 'prediction_by_class':
                return d[key] / divisor
            else:
                return {key: elementwise_div(d[key], divisor) for key in d.keys()}

        return {key: process_key(key) for key in d.keys()}


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
