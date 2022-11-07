import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gumbel_sigmoid_sample(logits, temperature, eps=1e-20):
    uniform1 = torch.rand(logits.size()).to(DEVICE)
    uniform2 = torch.rand(logits.size()).to(DEVICE)
    noise = -torch.log(torch.log(uniform2 + eps) / torch.log(uniform1 + eps) + eps)
    y = logits + noise
    return torch.sigmoid(y / temperature)


class DiaMultiClass(nn.Module):
    def __init__(self, args, cfg):
        super(DiaMultiClass, self).__init__()
        self.args = args
        self.cfg = cfg
        self.dropout = nn.Dropout(p=args.dropout)
        self.a_dim = cfg.a_dim

        self.net = nn.Sequential(nn.Linear(cfg.s_dim, cfg.h_dim),
                                 nn.ReLU(),
                                 nn.Linear(cfg.h_dim, cfg.h_dim),
                                 nn.ReLU(),
                                 nn.Linear(cfg.h_dim, cfg.a_dim))

        self.reset_param()
        self.loss = nn.BCEWithLogitsLoss()

    def reset_param(self):
        for part in [self.net]:
            for param in part.parameters():
                if param.dim() > 1:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.zeros_(param)

    def select_action(self, s):
        if self.args.gumbel:
            return gumbel_sigmoid_sample(self.net(s), 0.001).gt(0)
        else:
            return torch.sigmoid(self.net(s)).gt(0.5)

    def forward(self, s, a_target_gold, s_target_pos=None):
        """
        :param s_target_pos:
        :param s_target_gold: b * h_s where h_s is all 0 if not available
        :param curriculum:
        :param beta: prob to use teacher forcing
        :param a_target_gold: [b, 20]  [x, x, 171, x, x, x, 2, 0, 0, 0, 0, 0, 0]
        :param s: [b, s_dim]
        :return: hidden_state after several rollout
        """
        mask_cols = torch.LongTensor(range(self.cfg.max_len)).repeat(s.shape[0], 1).to(DEVICE)
        if len(s_target_pos.shape) == 1:
            s_target_pos = s_target_pos.unsqueeze(1)
        mask_begin = s_target_pos.repeat(1, self.cfg.max_len).to(DEVICE)
        mask = mask_cols.lt(mask_begin).long()

        probs = self.net(s)
        proc_tgt_tsr = torch.zeros(s.shape[0], self.a_dim).to(DEVICE)

        for i in range(self.cfg.max_len):
            temp_act_onehot = torch.zeros(s.shape[0], self.a_dim).to(DEVICE)
            eval_a_sample = a_target_gold[:, i].long().unsqueeze(1)
            src_tsr = torch.ones_like(eval_a_sample).float().to(DEVICE)
            temp_act_onehot.scatter_(-1, eval_a_sample, src_tsr)  # -- dim, index, val
            proc_tgt_tsr += temp_act_onehot * mask[:, i].unsqueeze(1)
            proc_tgt_tsr = proc_tgt_tsr.ge(1).float()

        loss_pred = self.loss(probs, proc_tgt_tsr)
        if self.args.gumbel:
            pred_act_tsr = gumbel_sigmoid_sample(probs, 0.001).gt(0)
        else:
            pred_act_tsr = torch.sigmoid(probs).ge(0.5)

        return loss_pred, pred_act_tsr


if __name__ == '__main__':
    args = {}
    args['dropout'] = 0.5
    args['gumbel'] = False


    # dict to class
    class Args:
        def __init__(self, entries):
            self.__dict__.update(entries)


    args = Args(args)

    cfg = {}
    cfg['s_dim'] = 553
    cfg['h_dim'] = 200
    cfg['a_dim'] = 166
    cfg['max_len'] = 20


    # dict to class
    class Cfg:
        def __init__(self, entries):
            self.__dict__.update(entries)


    cfg = Cfg(cfg)

    dia_multi_class = DiaMultiClass(args, cfg)
    s = torch.rand(32, 553)
    a_target_gold = torch.rand(32, 20)
    s_target_pos = torch.zeros(32, 1)
    r = dia_multi_class(s, a_target_gold, s_target_pos=s_target_pos)
    print(r)
