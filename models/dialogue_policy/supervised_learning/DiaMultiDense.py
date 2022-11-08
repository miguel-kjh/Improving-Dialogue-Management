import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dialogue_policy.supervised_learning.utils import GumbelConnector, onehot2id, id2onehot


class DiaMultiDense(nn.Module):
    def __init__(self, args, cfg):
        super(DiaMultiDense, self).__init__()
        self.args = args
        self.cfg = cfg
        self.use_gpu = torch.cuda.is_available()
        self.dropout = nn.Dropout(p=args.dropout)
        self.a_dim = cfg.a_dim

        self.net = nn.Sequential(nn.Linear(cfg.s_dim, cfg.h_dim),
                                           nn.ReLU(),
                                           nn.Linear(cfg.h_dim, cfg.a_dim//4),
                                           nn.LeakyReLU(0.2, True),)
        self.gumbel_length_index = self.cfg.a_dim * [2]
        self.gumbel_num = len(self.gumbel_length_index)
        self.last_layers = nn.ModuleList()
        self.gumbel_connector = GumbelConnector(False)
        for gumbel_width in self.gumbel_length_index:
            self.last_layers.append(nn.Linear(cfg.a_dim//4, gumbel_width))

        self.loss = nn.MultiLabelSoftMarginLoss()

    def select_action(self, s):
        h_s = self.net(s)
        input_to_gumbel = []
        for layer, g_width in zip(self.last_layers, self.gumbel_length_index):
            out = layer(h_s).unsqueeze(0)
            if self.args.gumbel:
                out = F.gumbel_softmax(out, dim=-1, tau=self.cfg.temperature)
                # out = self.gumbel_connector.forward_ST_gumbel(out.view(-1,  g_width), self.cfg.temperature)
            input_to_gumbel.append(out)
        action_rep = torch.cat(input_to_gumbel, -1)
        pred_act_tsr = onehot2id(action_rep)
        return pred_act_tsr

    def forward(self, s, a_target_gold, beta, s_target_gold=None, s_target_pos=None, train_type='train',  a_target_seq=None,a_target_full=None,a_target_pos=None):
        """
        :param s_target_pos:
        :param s_target_gold: b * h_s where h_s is all 0 if not available
        :param curriculum:
        :param beta: prob to use teacher forcing
        :param a_target_gold: [b, 20]  [x, x, 171, x, x, x, 2, 0, 0, 0, 0, 0, 0]
        :param s: [b, s_dim]
        :return: hidden_state after several rollout
        """
        mask_cols = torch.LongTensor(range(self.cfg.max_len)).repeat(s.shape[0], 1)
        if len(s_target_pos.shape) == 1:
            s_target_pos = s_target_pos.unsqueeze(1)
        mask_begin = s_target_pos.repeat(1, self.cfg.max_len)
        mask = mask_cols.lt(mask_begin).long()

        h_s = self.net(s)

        input_to_gumbel = []
        for layer, g_width in zip(self.last_layers, self.gumbel_length_index):
            out = layer(h_s)
            # if self.args.gumbel:
            # out = self.gumbel_connector.forward_ST_gumbel(out.view(-1, g_width), self.cfg.temperature)
            input_to_gumbel.append(out)
        action_rep = torch.cat(input_to_gumbel, -1)
        pred_act_tsr = onehot2id(action_rep)

        proc_tgt_tsr = torch.zeros(s.shape[0], self.a_dim)

        for i in range(self.cfg.max_len):
            temp_act_onehot = torch.zeros(s.shape[0], self.a_dim)
            eval_a_sample = a_target_gold[:, i].long().unsqueeze(1)
            src_tsr = torch.ones_like(eval_a_sample).float()
            temp_act_onehot.scatter_(-1, eval_a_sample, src_tsr)  # -- dim, index, val
            proc_tgt_tsr += temp_act_onehot * mask[:, i].unsqueeze(1)
            proc_tgt_tsr = proc_tgt_tsr.ge(1).float()
        loss_pred = self.loss(action_rep, id2onehot(proc_tgt_tsr))
        return torch.FloatTensor([0]), torch.zeros(s.shape[0], self.a_dim), \
               loss_pred, pred_act_tsr, \
               torch.FloatTensor([0]), torch.zeros_like(s), torch.FloatTensor([0]), \
               torch.FloatTensor([0]), torch.zeros(s.shape[0]), torch.zeros(s.shape[0])
