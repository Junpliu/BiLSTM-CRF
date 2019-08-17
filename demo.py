import torch
from model import BiLSTM_CRF
from utils import prepare_sequence, recognition_nameEntity

model_set = torch.load('model_set.pth')
EMBEDDING_DIM = model_set['EMBEDDING_DIM']
HIDDEN_DIM = model_set['HIDDEN_DIM']
word_to_ix = model_set['word_to_ix']
tag_to_ix = model_set['tag_to_ix']
all_tags_lst = model_set['all_tags_lst']

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, False)
check_point = torch.load('pretrained_model/37_0_054__885.pth.tar')
model.load_state_dict(check_point['state_dict'])
with torch.no_grad():
    while True:
        sentence = input('\nplease input the sentence: ').split()
        unknown = []
        for word in sentence:
            if word not in word_to_ix:
                unknown.append(word)
        for item in unknown:
            # print('unknown')
            sentence.remove(item)
        # print(sentence)
        precheck_sent = prepare_sequence(sentence, word_to_ix)
        output = model(precheck_sent)
        out_tag_lst = [all_tags_lst[tag_idx] for tag_idx in output[1]]
        # print(out_tag_lst)
        NE_tuple_lst = recognition_nameEntity(out_tag_lst)
        # print('NE_tuple_lst', NE_tuple_lst)

        for type in NE_tuple_lst:
            # print(item)
            if len(NE_tuple_lst[type]) != 0:
                print(type, ':', end='')
                for i, item in enumerate(NE_tuple_lst[type]):
                    if i != 0:
                        print('、', end='')
                    print(' '.join(sentence[item[0]:item[1]+1]), end='')
                print()