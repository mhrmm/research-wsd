Reed Word Sense Disambiguation
------------------------------

## Installing the reed_wsd package:
The repo only works when it is installed as a package

From the top-level directory:

    pip install -e .

## Preprocessing Data
Before doing any training, it is necessary to download and preprocess the data first.
### To download and preprocess the Raganato 2017 data:

From the top-level directory, run:

    cd reed_wsd/allwords
    bash ./install.sh

### To download and preprocess the imdb data:

From top-level directory, run:

    cd reed_wsd/imdb
    bash ./install.sh

### To Run Experiments
compile a list of configurations in a json file, and run
    
    python3 experiment.py path_to_config_file path_where_you_save_results

Possible configurations are listed below

    config =  {'task': 'mnist'/'allwords'/'imdb',
               'architecture': 'abstaining'/'simple'/'bem',
               'confidence': 'inv_abs'/'max_prob'/'max_non_abs',
               'criterion': {'name': 'pairwise'/'nll'/'crossentropy'/'conf1'/'conf4',
			     'alpha': float,
			     'warmup_epochs': int},
               'confused': bool,
               'style': 'pairwise'/'single',
               'dev_corpus': corpus_id,
               'bsz': int,
               'n_epochs': int
             }

### To locally install the reed_wsd package:

From the top-level directory:

    pip install -e .

### To run the official scoring script

The scorer ("Scorer.java") is provided by Raganato et al (2017).
To use the scorer, you first need to compile:

	javac allwords/data/WSD_Evaluation_Framework/Evaluation_Datasets/Scorer.java

Then, evaluate your system by typing the following commands: 

    cd allwords/data/WSD_Evaluation_Framework/Evaluation_Datasets
    java Scorer [gold-standard] [system-output]

Example of usage:

	cd allwords/data/WSD_Evaluation_Framework/Evaluation_Datasets
	java Scorer semeval2007/semeval2007.gold.key.txt output.txt

Please note that the official scoring programs are case sensitive and may be
sensitive if a different default character encoding is used.  The answer
format is the same of the gold-standard format. 

## Testing
### To run all unit tests

From the top-level directory, run: 
    
    cd allwords
    python3 -m unittest

### To run a particular unit test module (e.g. test/test_align.py)

From the top-level directory, run:

    cd allwords
    python3 -m unittest test.test_align
    
## DVC

If you pull from this repo now, `dvc` is already initialized.
You can download the data from remote storage using command

	dvc pull -r myremote

To check if there are any updates among your added data files, use command

	dvc status

To add a file or directory to your dvc cache, use command

	dvc add to_be_added

Adding or editing your added data files might update `.dvc`, `dvc.config` and other
files dvc uses to version your data, remember to add those files to your git commits
to record the versioning of data in your remote repo.

If you want to upload your data, use command

	dvc push -r myremote

For more details how `dvc remote` works, see [here](https://dvc.org/doc/command-reference/remote#remote).

    
