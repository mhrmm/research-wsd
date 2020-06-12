# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

class AffineClassifier(nn.Module): 
    
    def __init__(self, input_size, num_labels):
        super(AffineClassifier, self).__init__()
        print("input_size: {}; output_size: {}".format(input_size, num_labels))
        self.linear1 = nn.Linear(input_size, num_labels)
        torch.nn.init.xavier_uniform_(self.linear1.weight)

    def forward(self, input_vec):
        nextout = input_vec
        nextout = self.linear1(nextout)
        return nextout

class DropoutClassifier(nn.Module): 
 
    def __init__(self, input_size, hidden_size, num_labels):
        super(DropoutClassifier, self).__init__()
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(hidden_size, num_labels)

    def forward(self, input_vec):
        nextout = input_vec
        nextout = self.linear1(nextout).clamp(min=0)
        nextout = self.dropout1(nextout)
        nextout = self.linear2(nextout).clamp(min=0)
        nextout = self.dropout2(nextout)    
        nextout = self.linear3(nextout)
        return F.log_softmax(nextout, dim=1)


"""
from pytorch_transformers import BertModel

class BertForSenseDisambiguation(torch.nn.Module):
    def __init__(self, classifier=DropoutClassifier(1536, 100, 2)):
        super(BertForSenseDisambiguation, self).__init__()

        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = classifier
        self.count = 0

        
    def forward(self, input_ids_pair):
        print("Forward " + str(self.count))

        input_ids1, positions1 = input_ids_pair[:, 2:513], input_ids_pair[:, :1]
        input_ids2, positions2 = input_ids_pair[:, 513:], input_ids_pair[:, 1:2]
        #IPython.terminal.debugger.set_trace()
        final_layer1 = self.model(input_ids1.long())[0]
        #print("hrerse")
        final_layer2 = self.model(input_ids2.long())[0]

        index1 = torch.cat([positions1.unsqueeze(2)] * final_layer1.shape[2], dim=2)
        index2 = torch.cat([positions2.unsqueeze(2)] * final_layer2.shape[2], dim=2)

        vecs1 = final_layer1.gather(1, index1).squeeze(1)
        vecs2 = final_layer2.gather(1, index2).squeeze(1)

        vecs = torch.cat([vecs1, vecs2], dim=1)
        

        result = self.classifier(vecs)

        self.count += 1
        torch.cuda.empty_cache()

        return result
"""


