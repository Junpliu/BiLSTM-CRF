import os
class dataloader():
    def __init__(self, train_path, test_path):
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        self.tag_to_ix = {START_TAG: 0,
                     STOP_TAG: 1}  # all tags:['<START>', '<STOP>', 'B-ORG', 'O', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC']
        self.all_tags_lst = [START_TAG, STOP_TAG]
        self.training_data = []
        self.test_data = []

        self.processed_training = False  # first process training data.
        self.train_path = os.path.join('dataset', train_path)
        self.test_path = os.path.join('dataset', test_path)

        self.test_new_words = 0
    def getData(self):
        self._readtrainingfile(self.train_path)
        self._readtestfile(self.test_path)
        return self.training_data, self.test_data, self.all_tags_lst, self.tag_to_ix, self.word_to_ix

    def _readtrainingfile(self, filepath):
        with open(filepath, 'r') as f:
            sentences = f.read().split('\n\n')
            for sentence in sentences:
                if '-DOCSTART' in sentence:  # ignore the bordering symbol
                    continue
                word_lst = []
                tag_lst = []
                sentence_split = sentence.split('\n')
                if len(sentence_split) < 3:  # ignore the sentence having less than three words
                    continue
                for item in sentence_split:
                    item_split = item.split()
                    if not item_split:
                        continue
                    word_lst.append(item_split[0])
                    tag_lst.append(item_split[-1])
                    if item_split[-1] not in self.tag_to_ix:
                        self.tag_to_ix[item_split[-1]] = len(self.tag_to_ix)
                        self.all_tags_lst.append(item_split[-1])
                self.training_data.append((word_lst, tag_lst))
        print('all tags: ', self.all_tags_lst)
        self.word_to_ix = {}
        for sentence, tags in self.training_data:
            for word in sentence:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)

    def _readtestfile(self, filepath):
        with open(filepath, 'r') as f:
            sentences = f.read().split('\n\n')
            for sentence in sentences:
                if '-DOCSTART' in sentence:  # ignore the bordering symbol
                    continue
                word_lst = []
                tag_lst = []
                sentence_split = sentence.split('\n')
                if len(sentence_split) < 3:  # ignore the sentence having less than three words
                    continue
                # print('sentence: ', sentence)
                # input('')
                new_word_flag = False
                for item in sentence_split:
                    item_split = item.split()
                    if not item_split:
                        continue
                    if not item_split[0] in self.word_to_ix:  # new words
                        self.test_new_words += 1
                        new_word_flag = True
                        break
                    if item_split[-1] not in self.tag_to_ix:
                        print('found new tag! ')
                        input()
                    word_lst.append(item_split[0])
                    tag_lst.append(item_split[-1])
                # print('sentence after: ', word_lst)
                # input('')
                if not new_word_flag:
                    self.test_data.append((word_lst, tag_lst))
