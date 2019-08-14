from util import delete_last_line
import pprint
import pandas as pd
import os
import random
import seaborn as sns
from compare import getExampleSentencesBySense
import matplotlib.pyplot as plt
import torch
import json


#Note: 12 is the largest valid indice
def generate_random_layers_i(max_num_layers_to_average):
    """
    Randomly creates a list of layers to be averaged together. This function creates and
    returns a valid instance of the layers_i parameter from train_lemma_classifier_with_vec.
    Note it will also return a value for add_sent_encoding.
    """
    layers_to_average = random.randrange(1, max_num_layers_to_average+1)
    layers_i = []
    add_sent_encoding = False
    i = 0
    while i < layers_to_average:
        #If a value of 13 is generated it will be interpreted as add_sent_encoding=True
        layer = random.randrange(14)
        if layer == 13 and add_sent_encoding == False:
            add_sent_encoding = True
        elif not layer in layers_i:
            layers_i.append(layer)
        else:
            continue
        i += 1
    return layers_i, add_sent_encoding


def train_lemma_classifier_with_vec(layers_i, min_sense2_freq, max_sense2_freq, n_fold, max_sample_size, 
                                    verbose=True, add_sent_encoding=False):
    """
    train the lemma classifier with a specified vectorization patter

    layers_i is a list of indices that specifies the layers of output the vectorization is going to average
    add_sent_encoding is a boolean specifying whether including the sentence vector in the averaging

    For the rest of the parameters see train_lemma_classifiers in experiment.py
    """
    with open("bert.py", "a") as f:
        f.write("\nvectorize_instance = generate_vectorization(" + str(layers_i) + ", " +  str(add_sent_encoding) + ")")
    try:
        from experiment import train_lemma_classifiers
        lemma_info_dict = train_lemma_classifiers(min_sense2_freq, max_sense2_freq, n_fold, max_sample_size, verbose)
    finally:
        delete_last_line("bert.py")
    if add_sent_encoding == True:
            layers_i.append(True)
    return lemma_info_dict, layers_i

def train_lemma_classifier_with_vec_specific_lemmas(layers_i, lemmas_list, n_fold, max_sample_size, 
                                    verbose=True, add_sent_encoding=False):
    """
    train the lemma classifier with a specified vectorization patter

    layers_i is a list of indices that specifies the layers of output the vectorization is going to average
    add_sent_encoding is a boolean specifying whether including the sentence vector in the averaging

    For the rest of the parameters see train_lemma_classifiers in experiment.py
    """
    with open("bert.py", "a") as f:
        f.write("\nvectorize_instance = generate_vectorization(" + str(layers_i) + ", " +  str(add_sent_encoding) + ")")
    try:
        from experiment import train_lemma_classifiers_on_specific_lemmas
        lemma_info_dict = train_lemma_classifiers_on_specific_lemmas(lemmas_list, n_fold, max_sample_size, verbose)
    finally:
        delete_last_line("bert.py")
    if add_sent_encoding == True:
            layers_i.append(True)
    return lemma_info_dict, layers_i

def test_single_layer(min_sense2, max_sense2):
    layers = []
    for i in range(13):
        curr_dict = train_lemma_classifier_with_vec([i], min_sense2, max_sense2, 10, 600, verbose = False)
        accs = [lemma[0] for lemma in list(curr_dict.values())]
        layers.append(accs)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(layers)
    return layers
    
def train_lemma_classifier_with_diff_layers_specific_lemmas(max_layers_to_average, num_layer_combos, lemmas, n_fold, max_sample_size):
    """
    Calls train_lemma_classifier with a variety of values for layers_i and add_sent_encoding.
    It then writes the relevant data to a file.
    """
    layers_i_list = [ ([x], False) for x in range(13)]
    i = 0
    while i < num_layer_combos:
        layers_i_tuple = generate_random_layers_i(max_layers_to_average)
        if not layers_i_tuple in layers_i_list:
            layers_i_list.append(layers_i_tuple)
        else:
            continue
        i += 1
    
    data = []
    for layers_i_tuple in layers_i_list:
        print("hre")
        lemma_info_dict, layers_i = train_lemma_classifier_with_vec_specific_lemmas(layers_i_tuple[0], lemmas, n_fold, max_sample_size, 
                                        verbose=True, add_sent_encoding=layers_i_tuple[1])
        for key in lemma_info_dict.keys():
            lemma_info = lemma_info_dict[key]
            data.append([layers_i, key, lemma_info[0], lemma_info[1], lemma_info[2]])
    df = pd.DataFrame(data, columns=["spec", "lemma", "best_avg_acc", "sense1", "sense2"])

    num = 1
    while os.path.exists("classifier_data"+str(num)+".csv"):
        num += 1

    df.to_csv("classifier_data_spec"+str(num)+".csv", index=False)

def train_lemma_classifier_with_diff_layers(max_layers_to_average, num_layer_combos, min_sense2_freq, max_sense2_freq, n_fold, max_sample_size):
    """
    Calls train_lemma_classifier with a variety of values for layers_i and add_sent_encoding.
    It then writes the relevant data to a file.
    """
    layers_i_list = [ ([x], False) for x in range(13)]
    i = 0
    while i < num_layer_combos:
        layers_i_tuple = generate_random_layers_i(max_layers_to_average)
        if not layers_i_tuple in layers_i_list:
            layers_i_list.append(layers_i_tuple)
        else:
            continue
        i += 1
    
    data = []
    for layers_i_tuple in layers_i_list:
        lemma_info_dict, layers_i = train_lemma_classifier_with_vec(layers_i_tuple[0], min_sense2_freq, max_sense2_freq, n_fold, max_sample_size, 
                                        verbose=True, add_sent_encoding=layers_i_tuple[1])
        for key in lemma_info_dict.keys():
            lemma_info = lemma_info_dict[key]
            data.append([layers_i, key, lemma_info[0], lemma_info[1], lemma_info[2]])
    df = pd.DataFrame(data, columns=["spec", "lemma", "best_avg_acc", "sense1", "sense2"])

    num = 1
    while os.path.exists("classifier_data"+str(num)+".csv"):
        num += 1

    df.to_csv("classifier_data"+str(num)+".csv", index=False)

def store_csv_DF_from_lemma_classifier(lemma_info_dict, layers_i):
    """
    Takes the return values from the lemma_info_dict and converts them to 
    a pandas df before writing them to a file for later use.
    """
    
    data = []
    for key in lemma_info_dict.keys():
        lemma_info = lemma_info_dict[key]
        data.append([layers_i, key, lemma_info[0], lemma_info[1], lemma_info[2]])
    df = pd.DataFrame(data, columns=["spec", "lemma", "best_avg_acc", "sense1", "sense2"])
    
    num = 1
    while os.path.exists("classifier_data"+str(num)+".csv"):
        num += 1

    df.to_csv("classifier_datat"+str(num)+".csv", index=False)

def get_list_learnable_lemmas(good_threshold, file_name):
    """
    Returns a list containing the filenames of all the lemmas at or above the
    specified accuracy threshold in the specified data_file. 
    Note: The only valid datafiles are the classifier_data* files.
    """
    df = pd.read_csv(file_name)
    df = df[df["best_avg_acc"] >= good_threshold]
    df = df["lemma"]
    lemma_list = df.values.tolist()
    return lemma_list

def present_csv_DF_data(good_threshold, bad_threshold, num_example_sentences):
    """
    num_lemmas controls the total number of lemmas that will be included
    good_threshold controls the minimum average accuracy a lemma must have to 
    """
    
    df = pd.read_csv("classifier_data3.csv")
    sns.set(style="whitegrid")

    specs = sns.barplot(x="spec", y="best_avg_acc", data=df)
    plt.xticks(rotation=45)
    plt.show()

    lemmas = sns.barplot(x="lemma", y="best_avg_acc", data=df)
    plt.xticks(rotation=45)
    plt.show()


    df = pd.read_csv("classifier_data8_20-max.csv")
    
    best = df[df["best_avg_acc"] >= 0.7]
    worst = df[df["best_avg_acc"] <= 0.53]
    combined = pd.concat([best, worst])
    comb_lemmas = sns.barplot(x="lemma", y="best_avg_acc",data=combined)
    plt.xticks(rotation=45)
    plt.show()
    
    df_list = combined.values.tolist()

    def printSentences(sents):
        for sent in sents:
            print(sent)
        print()

    while True:
        lemma = input("\nenter a lemma to see its sentences:\n")
        for row in df_list:
            if row[1] == lemma:
                printSentences(getExampleSentencesBySense(row[3], 3))
                printSentences(getExampleSentencesBySense(row[4], 3))

#classifier_data_
#spec, lemma, best_avg_acc, sense1, sense2
def human_acc_test(threshold_high, threshold_low, filename, test_size):
    df = pd.read_csv(filename)
    high_acc_df = df[df["best_avg_acc"] >= threshold_high]
    low_acc_df = df[df["best_avg_acc"] <= threshold_low]
    cutoff = min(test_size, len(high_acc_df.index), len(low_acc_df.index))
    high_acc_df = high_acc_df.iloc[:cutoff]
    low_acc_df = low_acc_df.iloc[:cutoff]

    data = pd.concat([high_acc_df, low_acc_df]).sample(frac=1)
    with open("data/sense_to_pofs_dict.json") as f:
        sense_pos_dict = json.load(f)

    correct_count_high = 0
    correct_count_low = 0
    diff_pos_count_high = 0
    diff_pos_count_low = 0
    exports = []

    username = input("input your username: ")

    for i in data.index:
        acc = data["best_avg_acc"][i]
        lemma = data["lemma"][i]
        sense1 = data["sense1"][i]
        sense2 = data["sense2"][i]

        pos1 = sense_pos_dict[sense1]
        pos2 = sense_pos_dict[sense2]
        same_pos = True if pos1 == pos2 else False

        if not same_pos:
            if acc >= threshold_high:
                diff_pos_count_high += 1
            if acc <= threshold_low:
                diff_pos_count_low += 1

        sentences1 = getExampleSentencesBySense(sense1, 3)
        sentences2 = getExampleSentencesBySense(sense2, 3)

        is_same = True if random.random() >= 0.5 else False

        if is_same:
            one_or_two = True if random.random() >= 0.5 else False
            if one_or_two: pair = random.sample(sentences1, 2)
            else: pair = random.sample(sentences2, 2)
        else:
            pair = random.sample(sentences1, 1) + random.sample(sentences2, 1)


        correct_answer = is_same

        print(lemma + ": ")
        print("sentence 1: " + pair[0])
        print("sentence 2: " + pair[1])
        answer = input("Do the sentences have the same definition for the lemma word?(y/n) ")
        while not(answer == "y" or answer == "n"):
            print("answer input can only be y or n")
            answer = input("Do the sentences have the same definition for the lemma word?(y/n) ")
        answer = True if answer == "y" else False

        if_correct = True if answer == correct_answer else False

        export = [lemma, pair[0], pair[1], if_correct, sense1, sense2, is_same]
        exports.append(export)

        if if_correct:
            if acc >= threshold_high: correct_count_high += 1
            if acc <= threshold_low: correct_count_low += 1
    if len(high_acc_df.index) > 0:
        human_acc_high = correct_count_high / len(high_acc_df.index)
        diff_perc_high = diff_pos_count_high / len(high_acc_df.index)
    else:
        human_acc_high = 0
        diff_perc_high = 0
    if len(low_acc_df.index) > 0:
        human_acc_low = correct_count_low / len(low_acc_df.index)
        diff_perc_low = diff_pos_count_low / len(low_acc_df.index)
    else:
        human_acc_low = 0
        diff_perc_low = 0

    print("high accuracy stats:")
    print("percentage of different POS: {:.3f}".format(diff_perc_high))
    print("human disambiguation accuracy: {:.3f}".format(human_acc_high))

    print("low accuracy stats:")
    print("percentage of different POS: {:.3f}".format(diff_perc_low))
    print("human disambiguation accuracy: {:.3f}".format(human_acc_low))

    exports = pd.DataFrame(exports, columns = ["lemma", "sent1", "sent2", "if_correct", "sense1", "sense2", "is_same"])
    exports.to_csv("data/human_test_logs/" + username + ".csv", index=False)

    result_d = {}
    result_d["human_acc_high"] = human_acc_high
    result_d["diff_perc_high"] = diff_perc_high
    result_d["human_acc_low"] = human_acc_low
    result_d["diff_perc_low"] = diff_perc_low
    with open("data/human_test_results/" + username + ".json", "w") as f:
        json.dump(result_d, f)



        """
        for sent in sentences1:
            sentences.append((sent, sense1))
        for sent in sentences2:
            sentences.append((sent, sense2))

        random.shuffle(sentences)

        print(lemma + ": ")
        for j, sent in enumerate(sentences):
            print("sentence " + str(j) + ": " + sent[0])
        print("")

        response = input("There are two senses of the lemma, three of each. Type in the number label of the sentences that has the same sense. e.g. \"123\".\n")
        
        six = list(range(6))
        nums = []
        for c in response:
            nums.append(int(c))
        all_correct = True
        correct_sense = sentences[nums[0]][1]
        for n in nums:
            if_correct = True if sentences[n][1] == correct_sense else False
            if not if_correct: all_correct = False
            exports.append([username, lemma, sentences[n][0], if_correct, sentences[n][1]])
            six.remove(n)
        if correct_sense == sense1:
            correct_sense = sense2
        else:
            correct_sense = sense1
        for n in six:
            if_correct = True if sentences[n][1] == correct_sense else False
            exports.append([username, lemma, sentences[n][0], if_correct, sentences[n][1]])
        if data["best_avg_acc"][i] >= threshold_high:
            if all_correct:
                correct_count_high += 1
                print("correct!")
            else: print("wrong!")
            if not same_pos: diff_pos_count_high += 1
        if data["best_avg_acc"][i] <= threshold_low:
            if all_correct: 
                correct_count_low += 1
                print("correct!")
            else: print("wrong!")
            if not same_pos: diff_pos_count_low += 1

    exports = pd.DataFrame(exports, columns = ["user", "lemma", "sent", "if_correct", "sense"])
    exports.to_csv("data/" + username + ".csv", index=False)
    """












if __name__ == "__main__":
    #present_csv_DF_data(1,1,1)
    #train_lemma_classifier_with_diff_layers(1, 0, 20, 1000000, 10, 1000)
    """ l = get_list_learnable_lemmas(0.7, "classifier_data8_20-max.csv")
    print(len(l))
    lemma_info_dict, layers_i = train_lemma_classifier_with_vec_specific_lemmas([0],l, 1, 50)
    store_csv_DF_from_lemma_classifier(lemma_info_dict, layers_i)
    print("done") """
    lemmas = get_list_learnable_lemmas(0.7, "classifier_data8_20-max.csv")
    train_lemma_classifier_with_diff_layers_specific_lemmas(4, 1, lemmas, 10, 1000)
