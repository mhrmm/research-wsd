from reed_wsd.loss import NLLLoss, ConfidenceLoss1, PairwiseConfidenceLoss
from reed_wsd.mnist.networks import BasicFFN, AbstainingFFN
from reed_wsd.mnist.train import MnistDecoder, SingleTrainer, PairwiseTrainer
from reed_wsd.mnist.loader import MnistLoader, ConfusedMnistLoader, ConfusedMnistPairLoader
import torch
from os.path import join
import json
from torchvision import datasets, transforms
import os


file_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = join(file_dir, "data")
train_dir = join(data_dir, "train")
test_dir = join(data_dir, "test")
model_dir = join(file_dir, "saved")
validation_dir = join(file_dir, "validations")
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
if not os.path.isdir(validation_dir):
    os.mkdir(validation_dir)

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST(train_dir, download=True, train=True, transform=transform)
valset = datasets.MNIST(test_dir, download=True, train=False, transform=transform)
trainloader = ConfusedMnistLoader(trainset, bsz=64, shuffle=True)
valloader = ConfusedMnistLoader(valset, bsz=64, shuffle=True)

def baseline():
    criterion = NLLLoss()
    model = BasicFFN()#input_size: 784, hidden_size: [128, 64], output_size: 10
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    decoder = MnistDecoder()
    n_epochs = 20
    trainer = SingleTrainer(criterion, optimizer, trainloader, valloader, decoder, n_epochs)
    best_model = trainer(model)

def confidence_inv():
    criterion = ConfidenceLoss1(0.5)
    model = AbstainingFFN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    decoder = MnistDecoder()
    n_epochs = 20
    trainer = SingleTrainer(criterion, optimizer, trainloader, valloader, decoder, n_epochs)
    best_model = trainer(model)

def confidence_max():
    criterion = ConfidenceLoss1(0.5)
    model = AbstainingFFN(confidence_extractor='max_non_abs')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    decoder = MnistDecoder()
    n_epochs = 20
    trainer = SingleTrainer(criterion, optimizer, trainloader, valloader, decoder, n_epochs)
    best_model = trainer(model)

def confidence_twin():
    criterion = PairwiseConfidenceLoss()
    trainloader = ConfusedMnistPairLoader(trainset, bsz = 64, shuffle=True)
    valloader = ConfusedMnistLoader(valset, bsz = 64, shuffle=True)
    model = AbstainingFFN(confidence_extractor='abs')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    decoder = MnistDecoder()
    n_epochs = 20
    trainer = PairwiseTrainer(criterion, optimizer, trainloader, valloader, decoder, n_epochs)
    best_model = trainer(model)



def run(trainer, starting_model, name = None):
    if name is None:
        name = type(trainer.criterion).__name__
    print("================{} EXPERIMENT======================".format(name))
    net = trainer(starting_model)
    data_dict, _ = validate_and_analyze(net, trainer.val_loader)
    results_file = "{}.json".format(name.lower())
    with open(join(validation_dir, results_file), "w") as f:
        json.dump(data_dict, f)
    return net

if __name__ == "__main__":
    confidence_twin()
    
