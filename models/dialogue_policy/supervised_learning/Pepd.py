import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dialogue_policy.supervised_learning.utils import GumbelConnector, onehot2id, id2onehot
import random
import pytorch_lightning as pl

class Pepd(pl.LightningModule):
    def __init__(self, cfg):
        super(Pepd, self).__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.a_dim = cfg.a_dim
        self.decoder_hidden = cfg.h_dim // 2
        # State AutoEncoder
        self.state_encoder = nn.Sequential(nn.Linear(cfg.embed_size, cfg.h_dim),
                                           nn.ReLU(),
                                           self.dropout,
                                           nn.Linear(cfg.h_dim, self.decoder_hidden))

        self.state_reconstruction_head = nn.Sequential(nn.Linear(self.decoder_hidden, cfg.h_dim),
                                                       nn.ReLU(),
                                                       self.dropout,
                                                       nn.Linear(cfg.h_dim, cfg.s_dim))

        # Planning
        self.world_rnn = nn.GRU(cfg.embed_size, self.decoder_hidden, 1, bidirectional=False)
        self.act_emb = nn.Embedding(cfg.a_dim, cfg.embed_size)
        self.plan_head = nn.Sequential(nn.Linear(self.decoder_hidden, cfg.a_dim), )
        self.term_head = nn.Sequential(nn.Linear(self.decoder_hidden * 2, 2))
        # Prediction
        self.gumbel_length_index = self.cfg.a_dim * [2]
        self.gumbel_num = len(self.gumbel_length_index)
        self.last_layers = nn.ModuleList()
        self.gumbel_connector = GumbelConnector(False)
        for gumbel_width in self.gumbel_length_index:
            self.last_layers.append(
                nn.Sequential(nn.Linear(2 * self.decoder_hidden, cfg.a_dim // 4),
                              nn.ReLU(),
                              self.dropout,
                              nn.Linear(cfg.a_dim // 4, gumbel_width)))
        self.reset_param()
        self.CELoss = nn.CrossEntropyLoss(reduction='none')
        self.BCELoss = nn.BCEWithLogitsLoss()

    def reset_param(self):
        for part in [self.world_rnn]:
            for param in part.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.normal_(param.data)
        for part in [self.state_encoder, self.plan_head,
                     self.state_reconstruction_head, self.term_head, self.last_layers]:
            for param in part.parameters():
                if param.dim() > 1:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.zeros_(param)

    def select_action(self, s):
        s = s.unsqueeze(0)  # [1, s_dim]
        h_s = self.state_encoder(s)
        h_0 = h_s.clone()
        # MODULE: plan diverse paths
        h_decode_lst = []
        for _ in range(self.args.paths):
            h_s = h_0
            for _ in range(self.cfg.max_len):
                a_weights = self.plan_head(h_s)
                if self.args.gumbel:
                    a_sample = torch.argmax(F.gumbel_softmax(a_weights, dim=-1, tau=self.args.tau_plan_a),
                                            dim=-1).long()
                else:
                    a_sample = torch.argmax(a_weights, dim=-1).long()
                h_a = self.act_emb(a_sample)  # [1, a_emb]
                if self.args.residual:
                    residual = self.world_rnn(h_a.unsqueeze(0), h_s.unsqueeze(0))[1].squeeze(0)
                    h_s = h_s + residual
                else:
                    h_s = self.world_rnn(h_a.unsqueeze(0), h_s.unsqueeze(0))[1].squeeze(0)
                t_weights = self.term_head(torch.cat((h_0, h_s), dim=-1))
                if self.args.gumbel:
                    t_sample = torch.argmax(F.gumbel_softmax(t_weights, dim=-1, tau=self.args.tau_plan_t),
                                            dim=-1).long()
                else:
                    t_sample = torch.argmax(t_weights, dim=-1).long()
                if t_sample == 1:
                    break
            h_decode = torch.cat((h_0, h_s), dim=-1)
            h_decode_lst.append(h_decode)

        # Decoding:
        pair_wise_logits = []
        for layer, g_width in zip(self.last_layers, self.gumbel_length_index):
            out_lst = [layer(x) for x in h_decode_lst]
            out = torch.mean(torch.stack(out_lst, dim=0), dim=0)
            if self.args.gumbel:
                out = F.gumbel_softmax(out, dim=-1, tau=self.cfg.temperature)
            pair_wise_logits.append(out)
        return onehot2id(torch.cat(pair_wise_logits, -1))

    def forward(
            self,
            s,
            a_target_gold,
            beta,
            s_target_gold=None,
            s_target_pos=None,
            a_target_full=None,
            a_target_pos=None
    ):

        s_target_pos = s_target_pos.squeeze(1)
        a_target_pos = a_target_pos.squeeze(1)
        plan_act_tsr = torch.zeros(s.shape[0], self.a_dim).to(self.device)
        loss_state = torch.FloatTensor([0]).to(self.device)
        loss_kl = torch.FloatTensor([0]).to(self.device)
        pred_weight_lst = []
        plan_state_lst = []
        plan_term_weight_lst = []
        train_type = 'valid'
        if train_type == 'train' and random.uniform(0, 1) < beta:
            teacher_forcing = True
        else:
            teacher_forcing = False

        # generate mask where 1 if action 0 if pad
        mask_cols = torch.LongTensor(range(self.cfg.max_len)).repeat(s.shape[0], 1).to(self.device)
        mask_begin = s_target_pos.unsqueeze(1).repeat(1, self.cfg.max_len)
        term_target_gold = mask_cols.eq(mask_begin - 1).float()

        mask_cols = torch.LongTensor(range(self.cfg.max_len)).repeat(s.shape[0], 1).to(self.device)
        mask_begin = a_target_pos.unsqueeze(1).repeat(1, self.cfg.max_len)
        mask = mask_cols.lt(mask_begin).long()

        # ENCODE: initial state
        h_s = self.dropout(self.state_encoder(s))
        h_0 = h_s.clone()

        # LOSS: initial state reconstruction
        s_0_rec = self.state_reconstruction_head(h_s)
        loss_state += self.BCELoss(s_0_rec, s)

        # MODULE: planner train forward
        for step in range(self.cfg.max_len):
            plan_state_lst.append(h_s.unsqueeze(1))

            # PREDICT: current action
            a_weights = self.plan_head(h_s)
            pred_weight_lst.append(a_weights.unsqueeze(1))  # [b, 1, a_dim]

            # ENCODE: current gold action
            #
            a_sample = a_target_full[:, step].long()  # [b, 1]

            h_a = self.act_emb(a_sample)
            h_a = self.dropout(h_a)

            # TRANSITION: latent state
            if self.cfg.residual:
                residual = self.world_rnn(h_a.unsqueeze(0), h_s.unsqueeze(0))[1].squeeze(0)
                h_s = h_s + residual
            else:
                h_s = self.world_rnn(h_a.unsqueeze(0), h_s.unsqueeze(0))[1].squeeze(0)

            # PREDICT: current terminal
            plan_term_weight_lst.append(self.term_head(torch.cat((h_0, h_s), dim=-1)).unsqueeze(1))

            # EVALUATION: current action
            temp_act_onehot = torch.zeros(s.shape[0], self.a_dim).to(self.device)
            if self.cfg.gumbel:
                eval_a_sample = torch.argmax(F.gumbel_softmax(a_weights, dim=-1, tau=self.cfg.temperature),
                                             dim=-1).long().unsqueeze(1)
            else:
                eval_a_sample = torch.argmax(a_weights, dim=-1).long().unsqueeze(1)
            src_tsr = torch.ones_like(eval_a_sample).float()
            temp_act_onehot.scatter_(-1, eval_a_sample, src_tsr)  # -- dim, index, val
            plan_act_tsr += temp_act_onehot * mask[:, step].unsqueeze(1)
            plan_act_tsr = plan_act_tsr.ge(1).float()

        # LOSS: discrete action prediction
        pred_weight_mat = torch.cat(pred_weight_lst, dim=1)  # [b, L*topk, a_dim]
        loss_plan = self.CELoss(pred_weight_mat.contiguous().view(-1, pred_weight_mat.shape[2]),
                                a_target_full.contiguous().view(-1).long())
        loss_plan = torch.sum(loss_plan * mask.view(-1)) / torch.sum(mask)

        # LOSS: terminal prediction
        plan_term_weight_mat = torch.cat(plan_term_weight_lst, dim=1)
        loss_term = self.CELoss(plan_term_weight_mat.contiguous().view(-1, plan_term_weight_mat.shape[2]),
                                term_target_gold.contiguous().view(-1).long())
        loss_term = torch.sum(loss_term * mask.view(-1)) / torch.sum(mask)

        # EVALUATION: terminal
        plan_term_tsr = plan_term_weight_mat.argmax(-1).argmax(-1)  # [B]
        gold_term_tsr = term_target_gold.argmax(-1)

        # LOSS: next state prediction
        s_target_pos += (torch.arange(h_0.shape[0]) * len(plan_state_lst)).to(self.device)
        plan_state_concat = torch.cat(plan_state_lst, dim=1)  # -- [b * L, h_s] batch-0-time-1,2,3...,-batch-1-...
        plan_state_concat = torch.index_select(plan_state_concat.contiguous().view(-1, plan_state_concat.shape[-1]),
                                               dim=0, index=s_target_pos.long())
        h_s_rec = self.state_reconstruction_head(plan_state_concat)
        loss_state += self.BCELoss(h_s_rec, s_target_gold)

        # MODULE: plan diverse paths
        h_decode_lst = []
        for _ in range(self.cfg.paths):
            h_s = h_0
            plan_act_tsr = torch.zeros(s.shape[0], self.a_dim).to(self.device)
            plan_state_full = torch.zeros_like(h_s).to(self.device)
            noted_pos = torch.zeros(h_s.shape[0]).long().to(self.device)
            for _ in range(self.cfg.max_len):
                a_weights = self.plan_head(h_s)
                if self.cfg.gumbel:
                    a_sample = torch.argmax(F.gumbel_softmax(a_weights, dim=-1, tau=self.cfg.tau_plan_a),
                                            dim=-1).long()
                else:
                    a_sample = torch.argmax(a_weights, dim=-1).long()
                h_a = self.act_emb(a_sample)
                if self.cfg.residual:
                    residual = self.world_rnn(h_a.unsqueeze(0), h_s.unsqueeze(0))[1].squeeze(0)
                    h_s = h_s + residual
                else:
                    h_s = self.world_rnn(h_a.unsqueeze(0), h_s.unsqueeze(0))[1].squeeze(0)
                t_weights = self.term_head(torch.cat((h_0, h_s), dim=-1))
                if self.cfg.gumbel:
                    t_sample = torch.argmax(F.gumbel_softmax(t_weights, dim=-1, tau=self.cfg.tau_plan_t),
                                            dim=-1).long()
                else:
                    t_sample = torch.argmax(t_weights, dim=-1).long()
                noted_pos += (t_sample == 1)
                plan_state_full = plan_state_full * (noted_pos != 1).unsqueeze(1) + h_s * (noted_pos == 1).unsqueeze(1)
                noted_pos *= 2
                # EVALUATION: note current plan act
                temp_act_onehot = torch.zeros(s.shape[0], self.a_dim).to(self.device)
                src_tsr = torch.ones_like(a_sample).float()
                temp_act_onehot.scatter_(-1, a_sample.unsqueeze(1), src_tsr.unsqueeze(1))  # -- dim, index, val
                plan_act_tsr += temp_act_onehot * t_sample.unsqueeze(1)
                plan_act_tsr = plan_act_tsr.ge(1).float()
            if teacher_forcing:
                h_tgt = self.state_encoder(s_target_gold)
            else:
                h_tgt = plan_state_full
            h_decode = torch.cat((h_0, h_tgt), dim=-1)
            h_decode_lst.append(h_decode)
        # Decoding:
        pair_wise_logits, eval_logits = [], []
        for layer, g_width in zip(self.last_layers, self.gumbel_length_index):
            out_lst = [layer(x) for x in h_decode_lst]
            out = torch.mean(torch.stack(out_lst, dim=0), dim=0)
            if self.cfg.gumbel:
                eval_logits.append(F.gumbel_softmax(out, dim=-1, tau=self.cfg.temperature))
            else:
                eval_logits.append(out)
            pair_wise_logits.append(out)
        action_logits = torch.cat(pair_wise_logits, -1)
        pred_act_tsr = onehot2id(torch.cat(eval_logits, -1))

        # LOSS: macro-action prediction
        proc_tgt_tsr = torch.zeros(s.shape[0], self.a_dim).to(self.device)
        for i in range(self.cfg.max_len):
            temp_act_onehot = torch.zeros(s.shape[0], self.a_dim).to(self.device)
            eval_a_sample = a_target_gold[:, i].long().unsqueeze(1)
            src_tsr = torch.ones_like(eval_a_sample).float().to(self.device)
            temp_act_onehot.scatter_(-1, eval_a_sample, src_tsr)  # -- dim, index, val
            proc_tgt_tsr += temp_act_onehot * mask[:, i].unsqueeze(1)
            proc_tgt_tsr = proc_tgt_tsr.ge(1).float()
        loss_pred = self.BCELoss(action_logits, id2onehot(proc_tgt_tsr).to(self.device))

        return loss_pred, pred_act_tsr

if __name__ == '__main__':
    cfg = {}
    cfg['s_dim'] = 78
    cfg['h_dim'] = 200
    cfg['a_dim'] = 10
    cfg['max_len'] = 10
    cfg['dropout'] = 0.1
    cfg['embed_size'] = cfg['s_dim']
    cfg['residual'] = True
    cfg['gumbel'] = True
    cfg['temperature'] = 1e-3
    cfg['paths'] = 10
    cfg['tau_plan_a'] = 1e-3
    cfg['tau_plan_t'] = 1e-3


    # dict to class
    class Cfg:
        def __init__(self, entries):
            self.__dict__.update(entries)


    cfg = Cfg(cfg)
    model = Pepd(cfg)

    batch = 34
    s = torch.rand(batch, 78)
    a_target_gold = torch.tensor(
        [[0, 6, 6, 6, 6, 9, 9, 9, 9, 9]] * batch
    )
    s_target_pos = torch.tensor([[3] * batch]).T
    loss, pred = model.forward(
        s,
        a_target_gold,
        0.5,
        s,
        s_target_pos,
        a_target_gold,
        s_target_pos
    )
    print(loss)
    print(pred.size())
