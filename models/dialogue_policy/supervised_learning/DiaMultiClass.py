import torch
import torch.nn as nn


def gumbel_sigmoid_sample(logits, temperature, eps=1e-20):
    uniform1 = torch.rand(logits.size())
    uniform2 = torch.rand(logits.size())
    noise = -torch.log(torch.log(uniform2 + eps) / torch.log(uniform1 + eps) + eps)
    y = logits + noise
    return torch.sigmoid(y / temperature)


class DiaMultiClass(nn.Module):
    def __init__(self, cfg):
        super(DiaMultiClass, self).__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.a_dim = cfg.a_dim

        self.net = nn.Sequential(nn.LazyLinear(cfg.h_dim),
                                 nn.ReLU(),
                                 nn.Linear(cfg.h_dim, cfg.h_dim),
                                 nn.ReLU(),
                                 nn.Linear(cfg.h_dim, cfg.a_dim))

        #self.reset_param()
        self.loss = nn.BCEWithLogitsLoss()

    def reset_param(self):
        for part in [self.net]:
            for param in part.parameters():
                if param.dim() > 1:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.zeros_(param)

    def select_action(self, s):
        if self.cfg.gumbel:
            return gumbel_sigmoid_sample(self.net(s), 0.001).gt(0)
        else:
            return torch.sigmoid(self.net(s)).gt(0.5)

    def forward(self, s, a_target_gold, s_target_pos):
        max_len = a_target_gold.size(1)
        mask_cols = torch.LongTensor(range(max_len)).repeat(s.shape[0], 1)
        if len(s_target_pos.shape) == 1:
            s_target_pos = s_target_pos.unsqueeze(1)
        mask_begin = s_target_pos.repeat(1, max_len)
        mask = mask_cols.lt(mask_begin).long()

        probs = self.net(s)
        proc_tgt_tsr = torch.zeros(s.shape[0], self.a_dim)

        for i in range(max_len):
            temp_act_onehot = torch.zeros(s.shape[0], self.a_dim)
            eval_a_sample = a_target_gold[:, i].long().unsqueeze(1)
            src_tsr = torch.ones_like(eval_a_sample).float()
            temp_act_onehot.scatter_(-1, eval_a_sample, src_tsr)  # -- dim, index, val #TODO: check this
            proc_tgt_tsr += temp_act_onehot * mask[:, i].unsqueeze(1)
            proc_tgt_tsr = proc_tgt_tsr.ge(1).float()

        loss_pred = self.loss(probs, proc_tgt_tsr)
        if self.cfg.gumbel:
            pred_act_tsr = gumbel_sigmoid_sample(probs, 0.001).gt(0)
        else:
            pred_act_tsr = torch.sigmoid(probs).ge(0.5)

        return loss_pred, pred_act_tsr


if __name__ == '__main__':
    cfg = {}
    cfg['s_dim'] = 78
    cfg['h_dim'] = 200
    cfg['a_dim'] = 10
    cfg['max_len'] = 10
    cfg['dropout'] = 0.1
    cfg['gumbel'] = True


    # dict to class
    class Cfg:
        def __init__(self, entries):
            self.__dict__.update(entries)


    cfg = Cfg(cfg)

    dia_multi_class = DiaMultiClass(cfg)
    batch = 64
    s = torch.rand(batch, 78)
    a_target_gold = torch.rand(batch, 10)
    s_target_pos = torch.randint(0, 10, (batch, 1))
    r = dia_multi_class(s, a_target_gold, s_target_pos=s_target_pos)
    print(r)
