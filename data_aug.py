import random
import re
import nltk
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split

nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(nltk.corpus.stopwords.words('english'))


# Use a dictionary to store texts with the corresponding labels.
def data_textwithlabel(texts,labels):
    data_dic = {}
    for i in range(len(texts)):
            data_dic[texts[i]] = labels[i]
    
    return data_dic

# Randomly select 40% data of train dataset for data augmentation.
def select_data(data_dic):
    texts = list(data_dic.keys())
    number_data = len(texts) * 0.4
    return random.sample(texts, int(number_data))

# Randomly find a word and replace it with its synonym
def synonym_replacement(selected_data,data_dic):
    after_replace_data = {}
    for sentence in selected_data:
        new_sentence = sentence[:]
        random_words = [word for word in  re.findall(r'\w+', sentence) if word not in stop_words]
        chosen_words = random.sample(random_words, 2)
        for chosen_word in chosen_words:
            synonyms = find_synonyms(chosen_word)
            if len(synonyms) >= 1:
                synonym_word = random.choice(list(synonyms))
                new_sentence = new_sentence.replace(chosen_word,synonym_word)
        after_replace_data[new_sentence] = data_dic[sentence]
    return after_replace_data

# get sysnonyms of the word
def find_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word): 
	    for l in syn.lemmas():
             synonym = l.name().replace("_", " ").replace("-", " ").lower()
             synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
             synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)

    return list(synonyms)


# Randomly find two words and swap them
def random_word_swap(selected_data,data_dic):
    after_swap_data = {}
    for sentence in selected_data:
        random_words = [word for word in  re.findall(r'\w+', sentence) if word not in stop_words]
        chosen_words = random.sample(random_words, 2)
        words_list = re.findall(r"([\w']+|[^\w\s]+)", sentence)
        if chosen_words[0] in words_list and chosen_words[1] in words_list:
            index1 = words_list.index(chosen_words[0])
            index2 = words_list.index(chosen_words[1])
            temp = words_list[index1]
            words_list[index1] = words_list[index2]
            words_list[index2] = temp
            new_sentence = " ".join(words_list)
            after_swap_data[new_sentence] = data_dic[sentence]
    return after_swap_data

# Randomly insert a word to sentence
def word_insertion(selected_data,data_dic):
    after_insert_data = {}
    for sentence in selected_data:
        new_sentence = sentence[:]
        random_words = [word for word in  re.findall(r'\w+', sentence) if word not in stop_words]
        chosen_words = random.sample(random_words, 2)
        if chosen_words[0] in new_sentence :
            new_sentence = new_sentence.replace(chosen_words[0], "{} {}".format(chosen_words[1], chosen_words[0]),1)
            after_insert_data[new_sentence] = data_dic[sentence]
    return after_insert_data

# Randomly delete a word from sentence
def word_deletion(selected_data,data_dic):
    after_delete_data = {}
    for sentence in selected_data:
        random_words = [word for word in  re.findall(r'\w+', sentence) if word not in stop_words]
        chosen_words = random.sample(random_words, 2)
        words_list = re.findall(r"([\w']+|[^\w\s]+)", sentence)
        if chosen_words[0] in words_list and chosen_words[1] in words_list:
            words_list.remove(chosen_words[0])
            words_list.remove(chosen_words[1])
            new_sentence = " ".join(words_list)
            after_delete_data[new_sentence] = data_dic[sentence]
    return after_delete_data

# add augmented data with original data
def all_data_together(augment_data,original_data):
    texts = []
    labels = []
    for key in original_data:
        texts.append(key)
        labels.append(original_data[key])
    
    for key in augment_data:
        texts.append(key)
        labels.append(augment_data[key])

    return texts, labels

def data_augment(orignal_data,labels):
    data_dic = data_textwithlabel(orignal_data,labels)
    selected_data = select_data(data_dic)
    random.shuffle(selected_data)
    num_per_part = int(len(selected_data)/4)
    replace_data = synonym_replacement(selected_data[:num_per_part],data_dic)
    swap_data = random_word_swap(selected_data[num_per_part:num_per_part*2],data_dic)
    insert_data = word_insertion(selected_data[num_per_part*2:num_per_part*3],data_dic)
    delete_data = word_deletion(selected_data[num_per_part*3:],data_dic)
    augment_data = {**replace_data,**swap_data,**insert_data,**delete_data}
    return all_data_together(augment_data,data_dic)

