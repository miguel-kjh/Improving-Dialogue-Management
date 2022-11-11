import operator
from queue import PriorityQueue

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

INT = 0
LONG = 1
FLOAT = 2
EOS = 2


class GumbelConnector(nn.Module):
    def __init__(self, use_gpu):
        super(GumbelConnector, self).__init__()
        self.use_gpu = use_gpu

    def sample_gumbel(self, logits, use_gpu, eps=1e-20):
        u = torch.rand(logits.size())
        sample = Variable(-torch.log(-torch.log(u + eps) + eps))
        sample = cast_type(sample, FLOAT, use_gpu)
        return sample

    def gumbel_softmax_sample(self, logits, temperature, use_gpu):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        eps = self.sample_gumbel(logits, use_gpu)
        y = logits + eps
        return F.softmax(y / temperature, dim=y.dim() - 1)

    def soft_argmax(self, logits, temperature, use_gpu):
        return F.softmax(logits / temperature, dim=logits.dim() - 1)

    def forward(self, logits, temperature=1.0, hard=False,
                return_max_id=False):
        """
        :param logits: [batch_size, n_class] unnormalized log-prob
        :param temperature: non-negative scalar
        :param hard: if True take argmax
        :param return_max_id
        :return: [batch_size, n_class] sample from gumbel softmax
        """
        y = self.soft_argmax(logits, temperature, self.use_gpu)
        # y = self.gumbel_softmax_sample(logits, temperature, self.use_gpu)
        _, y_hard = torch.max(y, dim=1, keepdim=True)
        if hard:
            y_onehot = cast_type(Variable(torch.zeros(y.size())), FLOAT, self.use_gpu)
            y_onehot.scatter_(1, y_hard, 1.0)
            y = y_onehot
        if return_max_id:
            return y, y_hard
        else:
            return y

    def forward_ST(self, logits, temperature=0.8):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.soft_argmax(logits, temperature, self.use_gpu)
        # y = self.gumbel_softmax_sample(logits, temperature, self.use_gpu)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y

    def forward_ST_gumbel(self, logits, temperature=0.8):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature, self.use_gpu)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y


def cast_type(var, dtype, use_gpu):
    if use_gpu:
        if dtype == INT:
            var = var.type(torch.cuda.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.cuda.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.cuda.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    else:
        if dtype == INT:
            var = var.type(torch.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    return var


def onehot2id(onehot_list):
    id_list = []
    print()
    print(onehot_list.size())
    bs, a_dim = onehot_list.shape
    newlist = onehot_list.view(-1)
    for i in range(0, len(newlist), 2):
        if newlist[i] >= newlist[i + 1]:
            id_list.append(0)
        else:
            id_list.append(1)
    return torch.FloatTensor(id_list).view(bs, a_dim // 2)


def id2onehot(id_list):
    one_hot = []
    sp = id_list.shape
    a_dim = sp[-1]
    if type(id_list) == torch.Tensor:
        id_list = id_list.view(-1).tolist()
    for id in id_list:
        if id == 0:
            one_hot += [1, 0]
        elif id == 1:
            one_hot += [0, 1]
        else:
            raise ValueError("id can only be 0 or 1, but got {}".format(id))
    return torch.FloatTensor(one_hot).view(-1, a_dim * 2)


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def beam_decode(decoder, head, emb, decoder_hidden, sos_, eos_, h_0):
    '''
    :param decoder_hidden: input tensor of shape [H] for start of the decoding
    :return: decoded_batch
    '''

    beam_width = 4
    topk = 1  # how many sentence do you want to generate

    # Start with the start of the sentence token
    with torch.no_grad():
        decoder_input = Variable(torch.LongTensor([sos_]))

    # Number of sentence to generate
    endnodes = []
    number_required = min((topk + 1), topk - len(endnodes))

    # starting node -  hidden vector, previous node, word id, logp, length
    node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
    nodes = PriorityQueue()
    id_counter = 0
    # start the queue
    nodes.put((-node.eval(), id_counter, node))
    id_counter += 1

    qsize = 1
    tot_step = 1
    # start beam search
    while True:
        # give up when decoding takes too long
        if qsize > 2000: break

        # fetch the best node
        score, _, n = nodes.get()
        decoder_input = n.wordid
        decoder_hidden = n.h
        tot_step += 1
        if (n.wordid.item() == eos_ and n.prevNode != None) or tot_step == 20:
            endnodes.append((score, n))
            # if we reached maximum # of sentences required
            if len(endnodes) >= number_required:
                break
            else:
                continue
        # decode for one step using decoder
        decoder_hidden = decoder(torch.cat((emb(decoder_input), h_0), dim=-1).unsqueeze(0),
                                 decoder_hidden.unsqueeze(0))[1].squeeze(0)

        # PUT HERE REAL BEAM SEARCH OF TOP
        log_prob, indexes = torch.topk(F.log_softmax(head(decoder_hidden), -1), beam_width)
        nextnodes = []
        # import ipdb; ipdb.set_trace()
        for new_k in range(beam_width):
            log_p = log_prob[0][new_k].item()
            node = BeamSearchNode(decoder_hidden, n, indexes[0][new_k].view(1), n.logp + log_p, n.leng + 1)
            score = -node.eval()
            nextnodes.append((score, node))

        # put them into queue
        for i in range(len(nextnodes)):
            score, nn = nextnodes[i]

            nodes.put((score, id_counter, nn))
            id_counter += 1

            # increase qsize
        qsize += len(nextnodes) - 1

    # choose nbest paths, back trace them
    if len(endnodes) == 0:
        endnodes = [nodes.get() for _ in range(topk)]

    utterance = []
    for score, n in sorted(endnodes, key=operator.itemgetter(0)):
        utterance.append(n.wordid)
        # back trace
        while n.prevNode != None:
            n = n.prevNode
            utterance.append(n.wordid)
        utterance = utterance[::-1]

    last_id = None
    filtered_uttr = []
    for id_tsr in utterance:
        if id_tsr == last_id:
            continue
        filtered_uttr.append(id_tsr)
        last_id = id_tsr

    return torch.cat([emb(x) for x in filtered_uttr], dim=0), [x.unsqueeze(1) for x in filtered_uttr]
