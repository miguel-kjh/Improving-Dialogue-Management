import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dialogue_policy.supervised_learning.utils import GumbelConnector, onehot2id, id2onehot


class DiaMultiDense(nn.Module):
    def __init__(self, cfg):
        super(DiaMultiDense, self).__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.a_dim = cfg.a_dim

        self.net = nn.Sequential(nn.LazyLinear(cfg.h_dim),
                                 nn.ReLU(),
                                 nn.Linear(cfg.h_dim, cfg.a_dim // 4),
                                 nn.LeakyReLU(0.2, True), )
        self.gumbel_length_index = self.cfg.a_dim * [2]
        self.gumbel_num = len(self.gumbel_length_index)
        self.last_layers = nn.ModuleList()
        self.gumbel_connector = GumbelConnector(False)
        for gumbel_width in self.gumbel_length_index:
            self.last_layers.append(nn.Linear(cfg.a_dim // 4, gumbel_width))

        self.loss = nn.MultiLabelSoftMarginLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def forward(self, s, a_target_gold, s_target_pos):
        mask_cols = torch.LongTensor(range(self.cfg.max_len)).repeat(s.shape[0], 1).to(self.device)
        if len(s_target_pos.shape) == 1:
            s_target_pos = s_target_pos.unsqueeze(1)
        mask_begin = s_target_pos.repeat(1, self.cfg.max_len).to(self.device)
        mask = mask_cols.lt(mask_begin).long().to(self.device)

        h_s = self.net(s)

        input_to_gumbel = []
        for layer, g_width in zip(self.last_layers, self.gumbel_length_index):
            out = layer(h_s)
            # if self.args.gumbel:
            # out = self.gumbel_connector.forward_ST_gumbel(out.view(-1, g_width), self.cfg.temperature)
            input_to_gumbel.append(out)
        action_rep = torch.cat(input_to_gumbel, -1)
        pred_act_tsr = onehot2id(action_rep)

        proc_tgt_tsr = torch.zeros(s.shape[0], self.a_dim).to(self.device)

        for i in range(self.cfg.max_len):
            temp_act_onehot = torch.zeros(s.shape[0], self.a_dim).to(self.device)
            try:
                eval_a_sample = a_target_gold[:, i].long().unsqueeze(1)
            except:
                break
            src_tsr = torch.ones_like(eval_a_sample).float().to(self.device)
            temp_act_onehot.scatter_(-1, eval_a_sample, src_tsr)  # -- dim, index, val
            proc_tgt_tsr += temp_act_onehot.to(self.device) * mask[:, i].unsqueeze(1).to(self.device)
            proc_tgt_tsr = proc_tgt_tsr.ge(1).float()
        loss_pred = self.loss(action_rep, id2onehot(proc_tgt_tsr).to(self.device))
        return loss_pred, pred_act_tsr
