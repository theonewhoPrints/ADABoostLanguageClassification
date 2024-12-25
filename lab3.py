import math
import sys
import pandas as pd
import random
import pickle
"""
Isaac Soares, Lab 3 
CSCI 331 
ADA Boost and Decision Tree Implementation 


"""


# writes to file for model for DT.
def log_to_file(message):
    model_file = sys.argv[4]
    log = message + "\n"
    # Append the new log to the file
    with open(model_file, 'ab') as file:
        pickle.dump(log, file)

#unweighted dataframe
def read_input(examples, featuress):
    features = []
    input_vectors = []
    labels = []
    attrbutez = []

    # Read features from file
    with open(featuress, 'r') as file0:
        for line in file0:
            features.append(line.strip())

    with open(examples, 'r') as file:
        for line in file:
            vals = line.strip().split('|')
            if len(vals) == 2:
                label = vals[0]
                attributes = vals[1]
                attrbutez.append(attributes)
                labels.append(label)
            else:
                print("Invalid line format")

    for attribute in attrbutez:
        r = []
        for feature in features:
            r.append(feature in attribute)  # True if feature is in attribute, else False
        input_vectors.append(r)


    df = pd.DataFrame(input_vectors, columns=features)
    df["label"] = labels

    # Weights for first run.
    num_rows = len(df)
    weight = 1.0 / num_rows if num_rows > 0 else 0.0
    df["weight"] = [weight] * num_rows

    return df

def read_input_unweighted(examples, featuress):
    features = []
    input_vectors = []
    labels = []
    attrbutez = []

    with open(featuress, 'r') as file0:
        for line in file0:
            features.append(line.strip())

    with open(examples, 'r') as file:
        for line in file:
            vals = line.strip().split('|')
            if len(vals) == 2:
                label = vals[0]
                attributes = vals[1]
                attrbutez.append(attributes)
                labels.append(label)
            else:
                print("Invalid line format")

    for attribute in attrbutez:
        r = []
        for feature in features:
            r.append(feature in attribute)
        input_vectors.append(r)

    df = pd.DataFrame(input_vectors, columns=features)
    df["label"] = labels

    return df


# Ada Boost used in train
def Ada_Boost(dataframe, K, hypothesisOut):
    wn = []  # keep track of the split and weight in each array and put array in data file.
    Attributes = []
    labels = []

    for i in range(K): #per each cycle and run on a weak learner
        ts = Decision_tree_stump(dataframe)
        tree, Attribute, label = ts[0], ts[1], ts[2]
        error = 0
        Updated_W = 0

        for mapping in tree:
            for new_label, (row_index, old_label) in mapping.items():
                # Check if the old label is different from the new label
                if old_label != new_label:
                    error += dataframe.at[row_index, 'weight']
                    Updated_W += dataframe.at[row_index, 'weight'] #sum used for NORMALIZATION LATER

        error = max(min(error, 1 - 1e-10), 1e-10) #prevent bad arithmetics
        update = error / (1 - error)
        if error > 0.5:
            break
        hyp_weight = 0.5 * math.log((1 - error) / error)
        wn.append(hyp_weight)  #hypothesis weight to assciate with learner.

        Attributes.append(Attribute)  # best_attribute
        labels.append(label)

        for mapping in tree:
            for new_label, (row_index, old_label) in mapping.items():

                row = dataframe.iloc[row_index]
                if old_label == new_label:
                    dataframe.at[row_index, 'weight'] = row['weight'] * update
                    Updated_W += dataframe.at[row_index, 'weight']

        for mapping in tree:
            for new_label, (row_index, old_label) in mapping.items():
                row = dataframe.iloc[row_index]
                dataframe.at[row_index, 'weight'] = row['weight']/Updated_W

        i += 1

    with open(hypothesisOut, "wb") as file: #write too model.
        for i in range(len(wn)):
            # Access elements by index and write them to the file
            model = (f"{wn[i]}, {Attributes[i]}, {labels[i]}\n")
            pickle.dump(model, file)


#weak learner used for ADA Boost.
def Decision_tree_stump(dataframe, depth=0):
    mappings = []  # To store the label mappings for each row
    choices = ['en', 'nl']

    if len(dataframe['label'].unique()) == 1:
        return mappings, None, dataframe['label'].iloc[0]
    if len(dataframe.columns) == 2:
        label_counts = dataframe['label'].value_counts()
        majority_label = label_counts.idxmax() if label_counts.max() != label_counts.min() else random.choice(
            choices)
        return mappings, None, majority_label
    if dataframe.empty:
        label_counts = dataframe['label'].value_counts()
        majority_label = label_counts.idxmax() if label_counts.max() != label_counts.min() else random.choice(
            choices)
        return mappings, None, majority_label

    store_remainder = [] #remainders to compare to see best attribute.

    for attribute in dataframe.columns[:-2]: #per each attribute (disregard weight and label row)
        unique_values = dataframe[attribute].unique()
        total_entropy = 0
        for value in unique_values:
            value_rows = dataframe[dataframe[attribute] == value]
            label_counts = value_rows['label'].value_counts()
            value_entropy = 0
            for label, count in label_counts.items():
                sumweight = value_rows[value_rows['label'] == label]['weight'].sum()
                if count > 0:
                    prob = sumweight / len(value_rows) #consider weight instead

                    value_entropy += prob * math.log2(1 / prob)
            weighted_entropy = (value_rows['weight'].sum() / dataframe['weight'].sum()) * value_entropy

            total_entropy += weighted_entropy
        store_remainder.append(total_entropy)

    best_attribute = dataframe.columns[:-2][store_remainder.index(min(store_remainder))]

    best_attribute_values = dataframe[best_attribute].unique()

    majority_label_for_best = None  # Initialize the majority label for the best attribute
    major = []
    for value in best_attribute_values:
        subset_df = dataframe[dataframe[best_attribute] == value]
        if subset_df.empty:
            label_counts = dataframe['label'].value_counts()
            majority_label = label_counts.idxmax() if label_counts.max() != label_counts.min() else random.choice(
                choices)
            major.append(majority_label)
        else:
            label_counts = subset_df['label'].value_counts()
            majority_label = label_counts.idxmax() if label_counts.max() != label_counts.min() else random.choice(
                choices)

            for _, row in subset_df.iterrows():
                mappings.append({majority_label: (row.name, row['label'])})
            if majority_label_for_best is None:
                majority_label_for_best = majority_label

    return mappings, best_attribute, majority_label_for_best

# DT used for DT Training.
def Decision_tree(dataframe, depth=0, max_depth=3, used_attributes=None):
    choices = ['en', 'nl']
    if used_attributes is None:
        used_attributes = [] #have to keep tracked of attributes used so no repeat on Lt and Rt branches.

    # Check if the maximum depth is reached
    if depth >= max_depth:
        label_counts = dataframe['label'].value_counts()
        # random if labels amount same on ending branch.
        majority_label = label_counts.idxmax() if not label_counts.empty else random.choice(choices)
        log_to_file("  " * depth + f"{majority_label}") #dynamic logging while analysis being done.
        return
    # Check stopping conditions
    if len(dataframe['label'].unique()) == 1:
        log_to_file("  " * depth + f"{dataframe['label'].iloc[0]}")
        return
    if len(dataframe.columns) == 1 or dataframe.empty:
        label_counts = dataframe['label'].value_counts()
        majority_label = label_counts.idxmax() if not label_counts.empty else random.choice(choices)
        log_to_file("  " * depth + f"{majority_label}")
        return
    store_remainder = []
    for attribute in dataframe.columns[:-1]:
        if attribute in used_attributes:
            continue
        unique_values = dataframe[attribute].unique()
        total_entropy = 0

        for value in unique_values:
            value_rows = dataframe[dataframe[attribute] == value]
            label_counts = value_rows['label'].value_counts()

            value_entropy = 0
            for count in label_counts:
                if count > 0:
                    prob = count / len(value_rows)
                    value_entropy += prob * math.log2(1 / prob)

            weighted_entropy = (len(value_rows) / len(dataframe)) * value_entropy
            total_entropy += weighted_entropy

        store_remainder.append((attribute, total_entropy))

    if not store_remainder:
        label_counts = dataframe['label'].value_counts()
        majority_label = label_counts.idxmax() if not label_counts.empty else random.choice(choices)
        log_to_file("  " * depth + f"{majority_label}")
        return


    best_attribute, _ = min(store_remainder, key=lambda x: x[1])
    best_attribute_values = dataframe[best_attribute].unique()

    for value in best_attribute_values:
        subset_df = dataframe[dataframe[best_attribute] == value]
        if subset_df.empty:
            label_counts = dataframe['label'].value_counts()
            majority_label = label_counts.idxmax() if not label_counts.empty else random.choice(choices)
            log_to_file("  " * (depth + 1) + f"{majority_label}")
            continue

        log_to_file("  " * (depth + 1) + f"{best_attribute} = {value}")
        Decision_tree(
            subset_df.drop(columns=[best_attribute]),
            depth=depth + 1,
            max_depth=max_depth,
            used_attributes=used_attributes + [best_attribute]
        )


# line array used to split and clean data to access sentences to evaluate for predictions.
def linearr(examples):
    with open(examples, "r") as file:

        lines_array = [line.strip() for line in file.readlines() if line.strip()]
    return lines_array


def predict(examples, features, hypothesis):

    lines_array = linearr(examples)
    with open(hypothesis, 'rb') as file, open("hypotx.txt", 'w') as txt_file:
        try:
            while True:
                model = pickle.load(file)
                txt_file.write(model)  # putting from model into hypotx.txt
        except EOFError:
            pass  # End of file

    with open("hypotx.txt", "r") as file: #now using new txt file to find links when going down the tree.
        lines = [line.strip() for line in file if line.strip()]
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if '=' in line:
            key, value = map(str.strip, line.split('='))
            combined_key = f"{key}{value.capitalize()}"

            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line == 'en' or next_line == 'nl':
                    result.append({combined_key: next_line})
                    i += 1
                else:
                    if '=' in next_line:
                        next_key, _ = map(str.strip, next_line.split('='))
                        result.append({combined_key: next_key})
        i += 1

    # Function to find the next key
    def find_next_key(current_key, key_index):
        # Find the next kv in dict.
        # from range of key index so we don't traverse back
        for pair_index in range(key_index, len(result)):
            if current_key in result[pair_index]:
                # Update key_index to the current position
                key_index = pair_index
                return result[pair_index][current_key], key_index
        return None
    for line in lines_array:
        current_key = None
        next = None
        while next == None:
            if not current_key:
                pair = result[0]
                key = list(pair.keys())[0]
                key_index = 0
            else:
                key = current_key
            if key.endswith("True"):
                base_key = key[:-4]  # remove suffix's to eval. presence in setence.
            elif key.endswith("False"):
                base_key = key[:-5]
            else:
                base_key = key

            if not current_key:
                current_key = f"{base_key}True" if base_key in line else f"{base_key}False"
            v = find_next_key(current_key, key_index)
            next_value, key_index = v[0], v[1]
            if not next_value:
                break  # Stop processing if there's no next value
            if next_value in {"en", "nl"}:
                print(next_value)
                next = True
            elif next_value in line:
                current_key = f"{next_value}True"
            else:
                current_key = f"{next_value}False"


#predict based on a model from trained ADA Boost
def predict_ADA(examples, features, hypothesis):
    result = []
    sentences = linearr(examples) #sentences to evaluate
    with open(hypothesis, 'rb') as file, open("hypotx.txt", 'w') as txt_file:
        try:
            while True:
                model = pickle.load(file)
                txt_file.write(model)
        except EOFError:
            pass  # End of file reached

    try:
        with open("hypotx.txt", 'r') as file:
            for line in file:
                # Split the line by the comma which represents diff values.
                parts = line.strip().split(',', 1)
                if len(parts) == 2:  #both num & word.
                    try:
                        number = float(parts[0].strip())
                        word = parts[1].strip()
                        result.append({word: number})
                    except ValueError:
                        print(f"Skipping invalid number format in line: {line.strip()}")
    except FileNotFoundError:
        print("File not found.")
    sentence_values = []

    for sentence in sentences:
        sentence_value = 0
        for dictionary in result:
            for word, value in dictionary.items():
                define, lang = word.split(", ")

                if define in sentence:  # Check if the sentence contains the word (key)
                    # Ada Boost logic on determining value to add (ENG = +1 DUTCH = -1)
                    if lang == "en":
                        sentence_value += value
                    if lang == "nl":
                        value = -1 * value
                        sentence_value += value
                else:
                    if lang == "en":
                        value = -1 * value
                        sentence_value += value
                    if lang == "nl":
                        sentence_value += value
        # Append the final value of the sentence to the result array
        sentence_values.append(sentence_value)
    for weights in sentence_values:
        if weights < 0:
            print("nl")
        else:
            print("en")

def main():
    app = sys.argv[1]
    if app == "train":
        examples = sys.argv[2]
        featuress = sys.argv[3]
        hypothesisOut = sys.argv[4]
        learningtype = sys.argv[5]
        if learningtype == "ada":
            dataframe = read_input(examples, featuress)
            Ada_Boost(dataframe, 7, hypothesisOut)
        elif learningtype == "dt":
            dataframe = read_input_unweighted(examples, featuress)
            open(hypothesisOut, 'w').close()  # Clear the file at the start
            Decision_tree(dataframe)
        else:
            return 0
    elif app == "predict":
        examples = sys.argv[2]
        features = sys.argv[3]
        hypothesis = sys.argv[4]
        open("hypotx.txt", 'w').close()
        with open(hypothesis, 'rb') as file, open("hypotx.txt", 'w') as txt_file:
            try:
                while True:
                    model = pickle.load(file)
                    txt_file.write(model)
            except EOFError:
                pass
        with open("hypotx.txt", "r") as file:
            content = file.read(10).strip()
        if not content or (content[0].isdigit()):
            predict_ADA(examples, features, hypothesis)
            open("hypotx.txt", 'w').close()
        else:
            predict(examples, features, hypothesis)
            open("hypotx.txt", 'w').close()
    else:
        return 0

main()


