from utils import AverageMeter, prepare_sequence, evaluation
from model import *
import torch.optim as optim
import time
from getData import dataloader
import os
from optparse import OptionParser

EMBEDDING_DIM = 25
HIDDEN_DIM = 100
parser = OptionParser()
parser.add_option("-c","--checkpoint", dest="checkpoint",help="loading checkpoint", type="string")
(options,args)=parser.parse_args()

if options.checkpoint == None:
    check_point = ''# null str means training a model from scratch
else:
    check_point = os.path.join('pretrained_model/', options.checkpoint) # continue to train from the checkpoint
if check_point != '':
    print('checkpoint = ', check_point)

train_mode = True
GPU_available = False

dataset = dataloader('train.txt', 'test.txt')
training_data, test_data, all_tags_lst, tag_to_ix, word_to_ix = dataset.getData()
print('training data size: ', len(training_data), '\ntest data size: ', len(test_data))

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
    print('loading checkpoint: start_epoch{0} best_loss{1}'.format(start_epoch, best_loss))
else:
    start_epoch = 0
    best_loss = 100000000
    print('start training!')
F_value_best = 0


if GPU_available:
    model.cuda()
    print('moved model to GPU!!!')

end = time.time()
eps = 0.00000000001
for epoch in range(start_epoch, 100):
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