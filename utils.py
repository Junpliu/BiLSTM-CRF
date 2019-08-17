import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

# calculate F-value

def evaluation(output, truth):
    output_NE = set(compose_NE(output))
    truth_NE = set(compose_NE(truth))
    intersection = len(output_NE & truth_NE)
    # precision = intersection / (len(output_NE) + eps)
    # recall = intersection / (len(truth_NE) + eps)
    return intersection, len(truth_NE) - intersection, len(output_NE) - intersection

# compose the recognized named entity to compute F-value
def compose_NE(tag_lst):
    namedEntity = []
    cur_tag = 'O'
    cur_start = 0
    #cur_lst = ''
    for idx in range(len(tag_lst)):
        if tag_lst[idx].startswith('B'):
            cur_tag = tag_lst[idx]
            cur_start = idx
            #cur_lst += str(idx)
        elif tag_lst[idx].startswith('I'):
            if not (cur_tag.startswith('B') and cur_tag[-4] == tag_lst[idx][-4]):
                cur_tag = 'O'
                cur_start = idx
        else:
            if cur_tag != 'O':
                namedEntity.append(str(cur_start) + '-' + str(idx-1) + cur_tag)
                cur_tag = 'O'
        # print('cur_idx:', idx, 'cur_idx_tag:', tag_lst[idx], 'cur_tag: ', cur_tag, 'cur_start ', cur_start, 'namedEntity', namedEntity)
    if cur_tag != 'O':
        namedEntity.append(str(cur_start) + '-' + str(len(tag_lst)-1) + cur_tag)
    return namedEntity

def recognition_nameEntity(tag_lst):
    namedEntity = compose_NE(tag_lst)
    ret = {
        'ORG': [],
        'LOC': [],
        'MISC': [],
        'PER': []
    }
    for item in namedEntity:
        # print(item)
        ret[item.split('B')[1][1:]].append( (int(item.split('B')[0].split('-')[0]), int(item.split('B')[0].split('-')[1])) )
    return ret