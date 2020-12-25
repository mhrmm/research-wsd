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
    config_root = "mnist.abstain"
    config_ext = ".config.json"
    config_path = os.path.join("research-wsd/config", config_root + config_ext)
    output_path = os.path.join(MODELS_DIR, config_root + ".results.json")

    exp_seq = ExperimentSequence.from_json(config_path)
    result_db = exp_seq.run()
    result_db.save(output_path)
    reloaded = ResultDatabase.load(output_path)
    reloaded.results[0].show_training_dashboard()
