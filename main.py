import os
import subprocess
import sys

MODELS_BASE_DIR = os.getenv('REED_NLP_MODELS').strip()
if not os.path.isdir(MODELS_BASE_DIR):
    os.mkdir(MODELS_BASE_DIR)
MODELS_DIR = os.path.join(MODELS_BASE_DIR, 'research-wsd')
if not os.path.isdir(MODELS_DIR):
    os.mkdir(MODELS_DIR)

if __name__ == "__main__":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e",
                           "research-wsd/"])
    from reed_wsd.experiment import ExperimentSequence
    config_path = "research-wsd/data/abstain.mnist.config.json"
    output_path = os.path.join(MODELS_DIR, "foo.pt")

    exp_seq = ExperimentSequence.from_json(config_path)
    exp_seq.run_and_save(output_path)



