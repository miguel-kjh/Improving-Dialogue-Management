import torch
from torch.autograd import Variable
import torch.nn as nn
from models.dialogue_policy.supervised_learning.utils import beam_decode
import logging


class DiaSeq(nn.Module):
    def __init__(self, cfg):
        super(DiaSeq, self).__init__()
        self.cfg = cfg
        self.decoder_hidden = cfg.h_dim // 2
        self.net = nn.Sequential(nn.LazyLinear(cfg.h_dim),
                                 nn.ReLU(),
                                 nn.Linear(cfg.h_dim, self.decoder_hidden))
        self.act_emb = nn.Embedding(cfg.a_dim + 3, cfg.embed_size, padding_idx=cfg.a_dim + 2)
        self.decoder = nn.GRU(
            cfg.embed_size + self.decoder_hidden,
            self.decoder_hidden,
            1,
            dropout=cfg.dropout,
            bidirectional=False
        )

        self.pred_head = nn.Linear(self.decoder_hidden, cfg.a_dim + 3)

        self.use_gpu = torch.cuda.is_available()

        self.dropout = nn.Dropout(p=cfg.dropout)

        self.tau = cfg.temperature
        self.loss = nn.CrossEntropyLoss(ignore_index=cfg.a_dim + 2)
        self.a_dim = cfg.a_dim + 3
        self.sos_id = cfg.a_dim
        self.eos_id = cfg.a_dim + 1

    def select_action(self, s):
        # [1, s_dim]
        s = s.unsqueeze(0)
        pred_act_tsr = torch.zeros(s.shape[0], self.a_dim)

        h_s = self.net(s)
        h_0 = h_s.clone()

        if self.cfg.beam:
            _, id_lst = beam_decode(self.decoder, self.pred_head, self.act_emb, h_s, self.sos_id, self.eos_id, h_0)
            for id_tsr in id_lst:
                # print(id_lst)
                src_tsr = torch.ones_like(id_tsr).float()
                pred_act_tsr.scatter_(-1, id_tsr, src_tsr)  # -- dim, index, val
        else:
            with torch.no_grad():
                bos_var = Variable(torch.LongTensor([self.sos_id]))
            a_sample = bos_var.expand(s.shape[0], 1)
            for step in range(self.cfg.max_len):
                h_a = self.act_emb(a_sample.squeeze(1))
                output, h_s = self.decoder(torch.cat((h_a, h_0), dim=-1).unsqueeze(0), h_s.unsqueeze(0))
                h_s = h_s.squeeze(0)
                output = output.squeeze(0)
                a_weights = self.pred_head(output)
                a_sample = torch.argmax(torch.nn.functional.
                                        gumbel_softmax(a_weights, dim=-1, tau=1e-3), dim=-1).unsqueeze(1).long()
                # a_sample = torch.argmax(a_weights, dim=-1).unsqueeze(1).long()

                # for evaluation
                src_tsr = torch.ones_like(a_sample).float()
                pred_act_tsr.scatter_(-1, a_sample, src_tsr)  # -- dim, index, val
                if a_sample == self.eos_id:
                    break

        return pred_act_tsr[:, :-3]

    def forward(self, s, train_type='train', a_target_seq=None):
        pred_act_seq = []
        pred_act_tsr = torch.zeros(s.shape[0], self.a_dim)
        pred_weight_lst = []
        beam_size = 1

        # -- state encoding
        h_s = self.net(s)
        h_0 = h_s.clone()

        # -- predicting
        with torch.no_grad():
            bos_var = Variable(torch.LongTensor([self.sos_id]))
        a_sample = bos_var.expand(s.shape[0] * beam_size, 1)

        # |h_0, h_t| for decoding init state
        h_s = self.dropout(h_s)
        for step in range(self.cfg.max_len):
            h_a = self.act_emb(a_sample.squeeze(1))
            h_a = self.dropout(h_a)
            output, h_s = self.decoder(torch.cat((h_a, h_0), dim=-1).unsqueeze(0), h_s.unsqueeze(0))
            h_s = h_s.squeeze()
            output = output.squeeze()
            a_weights = self.pred_head(output)
            pred_weight_lst.append(a_weights.unsqueeze(1))
            # import ipdb; ipdb.set_trace()
            if train_type == 'train':
                a_target = a_target_seq[:, step]
                a_sample = a_target.unsqueeze(1).long()
            else:
                a_sample = torch.argmax(a_weights, dim=-1).unsqueeze(1).long()

            # for evaluation
            eval_a_sample = torch.argmax(a_weights, dim=-1).unsqueeze(1).long()
            pred_act_seq.append(eval_a_sample)
            src_tsr = torch.ones_like(eval_a_sample).float()
            pred_act_tsr.scatter_(-1, eval_a_sample, src_tsr)  # -- dim, index, val

        # -- batch * len * h
        pred_weight_mat = torch.cat(pred_weight_lst, dim=1)
        loss_pred = self.loss(pred_weight_mat.contiguous().view(-1, pred_weight_mat.shape[2]),
                              a_target_seq.contiguous().view(-1).long())

        logging.debug('pred' + '-' * 10)
        logging.debug(torch.tensor([x[0].item() for x in pred_act_seq]))
        logging.debug(a_target_seq[0])

        return loss_pred, pred_act_tsr[:, :-3]


if __name__ == '__main__':
    cfg = {}
    cfg['s_dim'] = 78
    cfg['h_dim'] = 200
    cfg['a_dim'] = 10
    cfg['max_len'] = 10
    cfg['dropout'] = 0.1
    cfg['embed_size'] = 78
    cfg['beam'] = True
    cfg['temperature'] = 1e-3


    # dict to class
    class Cfg:
        def __init__(self, entries):
            self.__dict__.update(entries)


    cfg = Cfg(cfg)

    dia_multi_class = DiaSeq(cfg)
    batch = 34
    s = torch.rand(batch, 78)
    a_target_gold = torch.tensor(
        [[0, 6, 6, 6, 6, 9, 10, 10, 10, 10]] * batch
    )
    s_target_pos = torch.tensor([3] * batch)
    l, r = dia_multi_class(s, a_target_seq=a_target_gold)
    print(l)
    print(r)
