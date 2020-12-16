import os
import subprocess
import sys

if __name__ == "__main__":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e",
                           "research-wsd/"])
    from reed_wsd.experiment import ExperimentSequence
    config_path = "research-wsd/data/abstain.mnist.config.json"
    output_path = "foo.pt"

    exp_seq = ExperimentSequence.from_json(config_path)
    exp_seq.run_and_save(output_path)



