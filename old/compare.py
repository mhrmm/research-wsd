import torch
import pandas as pd
from pytorch_transformers import BertConfig, BertTokenizer, BertModel
import json
import unicodedata
import os
import string
import csv
import random
import numpy
from util import Cd
from itertools import product
import math
import operator

total_size = 0

def vectorizeWordInContext(sentence, pos, tokenizer, model):
    """
    take a word and its bert-style tokenized sentence to compute the vectorization in the context of a sentence.
    return the vector representation of the word
    """
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(sentence)).unsqueeze(0)
    outputs = model(input_ids)
    final_layer = outputs[0].squeeze(0)
    return final_layer[pos]


def breakToString(break_level):
    """
    This helper function returns the correct string to use when creating the
    natural sentence, based on the specified break level.
    """
    if break_level == "NO_BREAK" or break_level == "SENTENCE_BREAK":
        return ""
    else:
        return " "   

def getJsonSentences(data):
    """
    This function processes a single documents data. It scans through the words
    until it finds a sentence break and uses this to return a list of all the sentences
    in the document, where each sentence is a list of words.
    """
    sentence = []
    sentences = []
    for word in data:
        word["text"] = unicodeToAscii(word["text"])
        if word["break_level"] == "SENTENCE_BREAK":
            sentences.append(sentence)
            sentence = []
        if word["text"] != "":
            sentence.append(word)
    return sentences

def createLemmaData():
    with open("data/googledata.json") as f:
        file_data = json.load(f)

    with open("word_lemma_dict.json", "r") as f:
        word_lemma_dict = json.load(f)
    id_sent_dict = {}
    with Cd("lemmadata"): 
        sent_id = 0
        for document in file_data:
            doc_body = document["doc"]
            for sent_object in doc_body:
                words_with_sense = sent_object["senses"]
                id_sent_dict[sent_id] = sent_object["bert_sent"]
                tracking = sent_object["tracking"]
                print(sent_id)
                for word_object in words_with_sense:
                    if not word_object["word"] in word_lemma_dict:
                        continue
                    lemma = word_lemma_dict[word_object["word"]]
                    lemma_instance = {}
                    lemma_instance["sent_id"] = sent_id
                    tracking_pos = tracking.index(word_object["pos"])
                    lemma_instance["pos"] = tracking_pos
                    lemma_instance["sense"] = word_object["sense"]
                    lemma_instance["pofs"] = word_object["pofs"]
                    lemma_file_name = lemma+".json"
        
                    if os.path.exists(lemma_file_name):
                        with open(lemma_file_name, "r") as lemma_file:
                            lemma_instance_list = json.load(lemma_file)
                        lemma_instance_list.append(lemma_instance)
                        with open(lemma_file_name, "w") as lemma_file:
                            json.dump(lemma_instance_list, lemma_file)
                    else:
                        lemma_instance_list = [lemma_instance]
                        with open(lemma_file_name, "w") as lemma_file:
                            json.dump(lemma_instance_list, lemma_file)
                sent_id += 1
        with open("id_to_sent.json", "w") as id_to_sent_file:
            json.dump(id_sent_dict, id_to_sent_file)

def createLemmaData_elmo():
    with open('data/completedata.json') as f:
        file_data = json.load(f)

    with open("data/word_lemma_dict.json", "r") as f:
        word_lemma_dict = json.load(f)
    id_sent_dict = {}
    with Cd("lemmadata_elmo"): 
        sent_id = 0
        for document in file_data:
            doc_body = document["doc"]
            for sent_object in doc_body:
                words_with_sense = sent_object["senses"]
                id_sent_dict[sent_id] = sent_object["sent"]
                print(sent_id)
                for word_object in words_with_sense:
                    if not word_object["word"] in word_lemma_dict:
                        continue
                    lemma = word_lemma_dict[word_object["word"]]
                    lemma_instance = {}
                    lemma_instance["sent_id"] = sent_id
                    lemma_instance["pos"] = word_object["pos"] 
                    lemma_instance["sense"] = word_object["sense"]
                    lemma_file_name = lemma+".json"
        
                    if os.path.exists(lemma_file_name):
                        with open(lemma_file_name, "r") as lemma_file:
                            lemma_instance_list = json.load(lemma_file)
                        lemma_instance_list.append(lemma_instance)
                        with open(lemma_file_name, "w") as lemma_file:
                            json.dump(lemma_instance_list, lemma_file)
                    else:
                        lemma_instance_list = [lemma_instance]
                        with open(lemma_file_name, "w") as lemma_file:
                            json.dump(lemma_instance_list, lemma_file)
                sent_id += 1
            with open("id_to_sent.json", "w") as id_to_sent_file:
                json.dump(id_sent_dict, id_to_sent_file)

def createCsvData():
    config = BertConfig.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel(config)
    with Cd("lemmadata"):
        with open("id_to_sent.json") as sent_id_dict_file:
            sent_id_dict = json.load(sent_id_dict_file)
        for dir_item in os.listdir():
            if os.path.isfile(dir_item):
                if dir_item.endswith(".json") and dir_item != "id_to_sent.json":
                    print(dir_item)
                    with open(dir_item, "r") as f:
                        lemma_data = json.load(f)
                    with Cd("vectors"):
                        with open(dir_item[:-5]+".csv", "w") as vector_file:
                            writer = csv.writer(vector_file, delimiter=",")
                            for instance in lemma_data:
                                inst_sent_id = instance["sent_id"]
                                inst_sense = instance["sense"]
                                inst_sent = sent_id_dict[str(inst_sent_id)]
                                if(len(inst_sent) > 511):
                                    continue 
                                vector = vectorizeWordInContext(inst_sent, instance["pos"], tokenizer, model)
                                vec_list = vector.detach().tolist()
                                row_data = [inst_sent_id, instance["pos"], inst_sense] + vec_list
                                writer.writerow(row_data)
             




def getFormattedData(docnames):
    """
    This function converts all the specified documents into our second json format.
    """
    data = []
    with open("googledata.json") as json_file:
        data = json.load(json_file)

    formatted_data = []
    for document in data:
        if document["docname"] in docnames or "all" in docnames:
            doc_dict = {}
            doc_dict["docname"] = document["docname"]
            doc_dict["doc"] = []
            createSentenceDictionaries(document["doc"], doc_dict["doc"])
            formatted_data.append(doc_dict)
            print("finished processing the document: " + document["docname"])
    with open("completedataV2.json", "w") as json_file:
        json.dump(formatted_data, json_file, indent=4)

def getExampleSentencesBySense(sense, num_examples):
    with open("data/completedata.json") as data:
        file_data = json.load(data)   

    examples = []
    for document in file_data:
        doc = document["doc"]
        for sent_object in doc:
            for word_with_sense in sent_object["senses"]:
                if word_with_sense["sense"] == sense:
                    examples.append(sent_object["natural_sent"])
    if len(examples) >= num_examples:
        examples = random.sample(examples, num_examples)
    return examples



def createSentenceDictionaries(document_data, list_to_modify):
    """
    This helper function aids getFormattedData by creating the 'sentence' dictionaries
    and appending them to the sent formatted_data list.
    """
    sent_dict = {}
    for sent in getJsonSentences(document_data):
        natural_sent = ""
        raw_sent = []
        senses = []
        cur_pos = 0
        for word in sent:
            raw_sent.append(word["text"])
            natural_sent += breakToString(word["break_level"]) + word["text"]
            if "sense" in word:
                senses.append({"word": word["text"], "pos": cur_pos, "sense": word["sense"], "pofs": word["pos"]})
            cur_pos += 1
        bert_sent = getBertSentenceFromRaw(raw_sent)
        tracking = trackRawSentenceIndices(raw_sent, bert_sent)

        sent_dict = {}
        sent_dict["natural_sent"] = natural_sent
        sent_dict["sent"] = raw_sent
        sent_dict["bert_sent"] = bert_sent
        sent_dict["tracking"] = tracking
        sent_dict["senses"] = senses
        list_to_modify.append(sent_dict)
    return sent_dict


def trackRawSentenceIndices(raw_sent, bert_sent):
    """
    track the position each word in BERT tokenization belongs to in the original tokenization
    returns the tracking list
    """
    tracking = []
    n = 0
    i = 0 #keep track of the current word in bert_sent
    for raw_word in raw_sent:
        while i < len(bert_sent):
            if raw_word == "":
                break
            curr_bert = bert_sent[i]
            if curr_bert.startswith("##"):
                curr_bert = curr_bert[2:]
            tracking.append(n)
            raw_word = raw_word[len(curr_bert):]
            i += 1
        n += 1
    return tracking


def getBertSentenceFromRaw(raw_sent):
    """
    convert the original tokenization to the BERT tokenization
    returns the BERT tokenization
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_sent_list = []
    for raw_word in raw_sent:
        bert_tokens = tokenizer.tokenize(raw_word)
        bert_sent_list += bert_tokens
    return bert_sent_list


def unicodeToAscii(s):
    all_letters = string.ascii_letters + " .,;'-"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
    

def generateWordLemmaDict():
    with open("googledata.json", "r") as f:
        data = json.load(f)
    d = {}
    for doc in data:
        print("processing", doc["docname"], "...")
        for word in doc["doc"]:
            if "lemma" in word.keys() and "sense" in word.keys():
                d[word["text"]] = word["lemma"]
    with open("word_lemma_dict.json", "w") as f:
        json.dump(d, f, indent=4)

def sampleData(max_n_pairs=1000, limit_num_files_train=10, limit_num_files_test=2):
    t_train = []
    t_test = []
    with Cd("lemmadata/vectors"):
        files = os.listdir()
    rand_perm = list(range(len(files)))
    random.shuffle(rand_perm)
    indices_train = rand_perm[:limit_num_files_train]
    indices_test = rand_perm[limit_num_files_train: limit_num_files_train + limit_num_files_test]
    
    print("Training Data:")
    for i in indices_train:
        file = files[i]
        with Cd("lemmadata/vectors"):
            if not os.path.isfile(file): continue
        if file == "be.csv" or not file.endswith(".csv"): continue
        print("processing", file)
        curr = sampleTrainingDataFromFile(max_n_pairs, file).float()
        if curr.shape != torch.Size([0]):
            t_train.append(curr)

    print("Testing Data:")
    for j in indices_test:
        file = files[j]
        with Cd("lemmadata/vectors"):
            if not os.path.isfile(file): continue
        if file == "be.csv" or not file.endswith(".csv"): continue
        print("processing:", file)
        curr = sampleTrainingDataFromFile(max_n_pairs, file).float()
        if curr.shape != torch.Size([0]):
            t_test.append(curr)
    return (torch.cat(t_train), torch.cat(t_test))


def sampleTrainingDataFromFile(size, file):
    global total_size
    print(total_size)
    t = []
    with Cd("lemmadata/vectors"):
        data = pd.read_csv(file, delimiter=",")
        n_vectors = len(data.index)

        max_n_pairs = n_vectors * (n_vectors - 1) // 2

        p = product(list(range(n_vectors)), list(range(n_vectors)))
        pairs = []
        for pair in p:
            pairs.append(pair)
        random.shuffle(pairs)

        if size > max_n_pairs: size = max_n_pairs
        total_size += size

        k = 0
        while k < size:
            i, j = pairs.pop()
            instance1 = data.iloc[i]
            instance2 = data.iloc[j]
            if_same = 1 if instance1.iloc[2] == instance2.iloc[2] else 0
            if if_same == 1: i += 1
            else: j += 1
            pair = pd.concat([pd.Series([if_same]), instance1.iloc[3:], instance2.iloc[3:]])
            t.append(torch.from_numpy(numpy.float64(pair.values)))
            k += 1
    if len(t) == 0:
        result = torch.tensor([])
    else:
        result = torch.stack(t)
    return result

if __name__ == "__main__":
    createLemmaData_elmo("data/completedata.json")

