import torch
from transformers import BertModel, BertTokenizer


def tokenize_with_target(tokenizer, sent, target_i):
    """
    sent: a list of words to be tokenized
    target: the target word
    """
    sent = ['[CLS]'] + sent + ['[SEP]']
    tks = []
    start = 0
    tracker = 0
    for i, word in enumerate(sent):
        if i == target_i + 1:
            start = tracker
        curr_tks = tokenizer.tokenize(word)
        tracker += len(curr_tks)
        if i == target_i + 1:
            end = tracker
        tks += curr_tks
    return torch.tensor(tokenizer.convert_tokens_to_ids(tks)), [start, end]
    

class BertSentenceVectorizer:
    """
    A BertSentenceVectorizer produces a function that maps a sentence
    (represented as a string) to the (final-layer) BERT vectors for
    each of its tokens.
    
    """
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def __call__(self, sent):
        # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens 
        # in the right way for each model.
        input_ids = self.tokenizer.encode(sent, add_special_tokens=True) 
        bert_toks = self.tokenizer.convert_ids_to_tokens(input_ids)
        input_ids = torch.tensor([input_ids])
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]
            last_hidden_states = last_hidden_states.squeeze(0)
            return bert_toks, last_hidden_states
        
    def dim(self):
        return 768
    
    
class BertVectorizer:
    """
    A BertVectorizer produces a function that maps a SenseInstance to a
    BERT vector representation, by encoding the whole sentence and then
    returning the vector representation of the target token.
    
    """
    def __init__(self):
        self.vectorizer = BertSentenceVectorizer()
       
    def __call__(self, instance):
        bert_toks, states = self.vectorizer(' '.join(instance.tokens))
        return states[instance.pos]

    def dim(self):
        return self.vectorizer.dim()
