from utils import AverageMeter, prepare_sequence, evaluation
from model import *
import torch.optim as optim
import time
from getData import dataloader
import os

EMBEDDING_DIM = 25
HIDDEN_DIM = 100
check_point = os.path.join('pretrained_model/','37_0_054__885.pth.tar')#'36_0_057.pth.tar'
train_mode = False
GPU_available = False

print('check_point = ', check_point)

dataset = dataloader('train.txt', 'test.txt')
training_data, test_data, all_tags_lst, tag_to_ix, word_to_ix = dataset.getData()
print('training data size: ', len(training_data), '\ntest data size: ', len(test_data))

# training_data looks like:
# training_data = [(
#     "the wall street journal reported today that apple corporation made money".split(),
#     "B I I I O O O B I O O".split()
# ), (
#     "georgia tech is a university in georgia".split(),
#     "B I O O O O B".split()
# )]
GPU_available = GPU_available and torch.cuda.is_available()
print('GPU_available = ', GPU_available)
print('len_vocab = ', len(word_to_ix), 'tag_to_ix', tag_to_ix)

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, GPU_available)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

if check_point:
    check_point = torch.load(check_point)
    model.load_state_dict(check_point['state_dict'])
    optimizer.load_state_dict(check_point['optimizer'])
    best_loss = check_point['best_loss']
    start_epoch = check_point['epoch']
    print('loading checkpoint: start_epoch{0} best_loss{1}', start_epoch, best_loss)
else:
    start_epoch = 0
    best_loss = 100000000
F_value_best = 0
# with torch.no_grad():
#     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
#     precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
#     print('check output before training: ', [tags_lst[item] for item in model(precheck_sent)[1]])

if GPU_available:
    model.cuda()
    print('moved model to GPU!!!')
print('start training!')
end = time.time()
eps = 0.00000000001
for epoch in range(start_epoch, 1000):
    losses = AverageMeter()
    for i, (sentence, tags) in enumerate(training_data):
        model.zero_grad()

        # prepare torch.Tensor
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        if GPU_available:
            sentence_in = sentence_in.cuda()
            targets = targets.cuda()
        loss = model.neg_log_likelihood(sentence_in, targets)

        losses.update(loss.item(), 1)

        loss.backward()
        optimizer.step()

    if train_mode:
        print('Epoch: [{0}]\t'
            'Loss: {losses.avg:.4f}\t'
            'Time: {epoch_time:.2f}'.format(epoch, losses=losses, epoch_time=time.time()-end))

    end = time.time()
    # test process
    TP = 0
    FN = 0
    FP = 0
    with torch.no_grad():
        batch_time = AverageMeter()
        for i, (sentence, truth_tags_lst) in enumerate(test_data):
            # print('sentence :', sentence)
            precheck_sent = prepare_sequence(sentence, word_to_ix)
            output = model(precheck_sent)
            out_tag_lst = [all_tags_lst[tag_idx] for tag_idx in output[1]]
            # print('out_tag_lst = ', out_tag_lst)
            # print('truth_tag_lst = ', truth_tags_lst)
            tp, fn, fp = evaluation(out_tag_lst, truth_tags_lst)
            TP += tp
            FN += fn
            FP += fp
    # print(TP, FN, FP)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    F_value = (2 * precision * recall) / (precision + recall + eps)
    if F_value > F_value_best:
        F_value_best = F_value
    print('Test epoch [{0}]\t'
          'Time:{epoch_time:.2f}\t'
          'F-value:{value:.4f}'.format(epoch, epoch_time=time.time() - end, value=F_value))

    if train_mode and losses.avg < best_loss:
        best_loss = losses.avg
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            'best_F': F_value_best
        }
        torch.save(state, os.path.join('pretrained_model/', str(epoch) + '_' + str(best_loss)[:1] + '_' + str(best_loss)[2:5] + '__' + str(F_value)[2:5] + '.pth.tar'))
    end = time.time()