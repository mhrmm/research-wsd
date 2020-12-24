import os
import subprocess
import sys
from reed_wsd.experiment import ExperimentSequence
from reed_wsd.analytics import ResultDatabase

MODELS_BASE_DIR = os.getenv('REED_NLP_MODELS').strip()
if not os.path.isdir(MODELS_BASE_DIR):
    os.mkdir(MODELS_BASE_DIR)
MODELS_DIR = os.path.join(MODELS_BASE_DIR, 'research-wsd')
if not os.path.isdir(MODELS_DIR):
    os.mkdir(MODELS_DIR)

if __name__ == "__main__":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e",
                           "research-wsd/"])
    config_path = "research-wsd/config/mnist.abstain.config.json"
    output_path = os.path.join(MODELS_DIR, "mnist.abstain.results.json")

    exp_seq = ExperimentSequence.from_json(config_path)
    result_db = exp_seq.run()
    result_db.save(output_path)
    reloaded = ResultDatabase.load(output_path)
    reloaded.results[0].show_training_dashboard()

