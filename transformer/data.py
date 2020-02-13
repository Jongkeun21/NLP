from konlpy.tag import Okt
import pandas as pd
import tensorflow as tf
import enum
import os
import re
from sklearn.model_selection import train_test_split
import numpy as np
from configs import DEFINES
from tqdm import tqdm

PAD_MASK = 0
NON_PAD_MASK = 1

FILTERS = "([~.,!?\"':;)(])"
PAD = "<PADDING>"
STD = "<START>"
END = "<END>"
UNK = "<UNKNOWN>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)

def load_data() :
    data_df = pd.read_csv(DEFINES.data_path, header=0)
    question, answer = list(data_df['Q']), data_df['A']
    train_input, eval_input, train_label, eval_label = train_test_split(question, answer, test_size=0.33, random_state=42)
    
    return train_input, eval_input, train_label, eval_label

def prepro_like_morphlized(data) :
    morph_analyzer = Okt()
    result_data = list()
    
    for seq in tqdm(data) :
        # " ".join -> 리스트의 인덱스를 띄어쓰기로 구분하여 붙임 
        # example -- a = [1, 2, 3], " ".join(a) => 1 2 3
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphlized_seq)
        
    return result_data

def enc_processing(value, dictionary) :
    sequences_input_index = []
    sequences_length = []
    
    if DEFINES.tokenize_as_morph :
        value = prepro_like_morphlized(vale)
        
    for sequence in value :
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = []
        
        for word in sequence.split() :
            if dictionary.get(word) is not None :
                sequence_index.extend([dictionary[word]])
            else :
                sequence_index.extend([dictionary[UNK]])
        
        if len(sequence_index) > DEFINES.max_seq_length :
            sequence_index = sequence_index[:DEFINES.max_seq_length]
            
        sequences_length.append(len(sequence_index))
        sequence_index += (DEFINES.max_seq_length - len(sequence_index)) * [dictionary[PAD]]
        sequence_index.reverse()
        sequences_input_index.append(sequence_index)
        
    return np.asarray(sequences_input_index), sequences_length

def dec_target_processing(value, dictionary) :
    sequences_target_index = []
    sequences_length = []
    
    if DEFINES.tokenize_as_morph :
        value = prepro_like_morphlized(value)
        
    for sequence in value :
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = [dictionary[word] for word in sequence.split()]
        
        if len(sequence_index) >= DEFINES.max_seq_length :
            sequence_index = sequence_index[:DEFINES.max_seq_length-1] + [dictionary[END]]
        else :
            sequence_index += [dictionary[END]]
            
        sequences_length.append([PAD_MASK if num > len(sequence_index) else NON_PAD_MASK for num in range(DEFINES.max_seq_length)])
        sequence_index += (DEFINES.max_seq_length - len(sequence_index)) * [dictionary[PAD]]
        sequences_target_index.append(sequence_index)
        
    return np.asarray(sequences_target_index), np.asarray(sequences_length)

def pred2string(value, dictionary) :
    sentence_string = []
    
    if DEFINES.serving == True :
        for v in value['output'] :
            sentence_string = [dictionary[index] for index in v]
    else :
        for v in value :
            sentence_string = [dictionary[index] for index in v['indexs']]
            
    print("sentence_string: ", sentence_string)
    
    answer = ""
    
    for word in sentence_string :
        if word not in PAD and word not in END :
            answer += word
            answer += " "
            
    print("answer: ", answer)
    
    return answer

def rearrange(input, target) :
    features = {"input": input}
    
    return features, target

def train_rearrange(input, length, target) :
    features = {"input": input, "length": length}
    
    return features, target

def train_input_fn(train_input_enc, train_target_dec_length, train_target_dec, batch_size) :
    #tf.data.Dataset.from_tensor_slices(a, b) -- a와 b의 내용을 하나씩 자름
    dataset = tf.data.Dataset.from_tensor_slices((train_input_enc, train_target_dec_length, train_target_dec))
    dataset = dataset.shuffle(buffer_size=len(train_input_enc))
    
    assert batch_size is not None, "train batchSize must not be None"
    
    #batch(batch_size) -- dataset을 batch_size의 크기로 구분
    #map(func) -- func를 거친 data로 변환
    #repeat(cnt) -- data를 cnt의 횟수만큼 반복하여 저장
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(train_rearrange)
    dataset = dataset.repeat()
    
    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()

def eval_input_fn(eval_input_enc, eval_target_dec, batch_size) :
    dataset = tf.data.Dataset.from_tensor_slices((eval_input_enc, eval_taget_dec))
    dataset = dataset.shuffle(buffer_size=len(eval_input_enc))
    
    assert batch_size is not None, "eval batchSize must not be None"
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(rearrange)
    dataset = dataset.repeat(1)
    
    iterator = dataset.make_ont_shot_iterator()
    
    return iterator.get_next()

def data_tokenizer(data) :
    words = []
    
    for sentence in data :
        sentence = re.sub(CHANGE_FILTER, "", sentence)

        for word in sentence.split() :
            words.append(word)
            
    # word for word in words if word -- words 안의 index인 word가 True이면 word 반환(반복)
    return [word for word in words if word]    

def load_vocabulary() :
    vocabulary_list = []
    
    if (not (os.path.exists(DEFINES.vocab_path))) :
        if (os.path.exists(DEFINES.data_path)) :
            data_df = pd.read_csv(DEFINES.data_path, encoding='utf-8')
            question, answer = list(data_df['Q']), list(data_df['A'])
            
            if DEFINES.tokenize_as_morph :
                question = prepro_like_morphlized(question)
                answer = prepro_like_morphlized(answer)
                
            data = []
            data.extend(question)
            data.extend(answer)
            
            words = data_tokenizer(data)
            words = list(set(words))
            #[:0] -- 리스트 맨 앞에 '추가'할 때 사용하면 유용해
            words[:0] = MARKER
            
        with open(DEFINES.vocab_path, 'w', encoding='utf-8') as vocabulary_file :
            for word in words :
                vocabulary_file.write(word + '\n')
                
    with open(DEFINES.vocab_path, 'r', encoding='utf-8') as vocabulary_file :
        for line in vocabulary_file :
            #strip -- 문장 양쪽 끝의 공백 삭제(\n 포함)
            vocabulary_list.append(line.strip())
            
    char2idx, idx2char = make_vocabulary(vocabulary_list)
    
    return char2idx, idx2char, len(char2idx)

def make_vocabulary(vocabulary_list) :
    char2idx = {char: idx for idx, char in enumerate(vocabulary_list)}
    idx2char = {idx: char for idx, char in enumerate(vocabulary_list)}
    
    return char2idx, idx2char

def main(self) :
    char2idx, idx2char, vocabulary_length = load_vocabulary()

if __name__ == '__main' :
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)