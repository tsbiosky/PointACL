import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitsAdvLoss(nn.Module):

    def __init__(self, kappa=0.):
        """Adversarial function on logits.

        Args:
            kappa (float, optional): min margin. Defaults to 0..
        """
        super(LogitsAdvLoss, self).__init__()

        self.kappa = kappa

    def forward(self, logits, targets):
        """Adversarial loss function using logits.

        Args:
            logits (torch.FloatTensor): output logits from network, [B, K]
            targets (torch.LongTensor): attack target class
        """
        B, K = logits.shape
        if len(targets.shape) == 1:
            targets = targets.view(-1, 1)
        targets = targets.long()
        one_hot_targets = torch.zeros(B, K).cuda().scatter_(
            1, targets, 1).float()  # to one-hot
        real_logits = torch.sum(one_hot_targets * logits, dim=1)
        other_logits = torch.max((1. - one_hot_targets) * logits -
                                 one_hot_targets * 10000., dim=1)[0]
        loss = torch.clamp(other_logits - real_logits + self.kappa, min=0.)
        return loss.mean()


class CrossEntropyAdvLoss(nn.Module):

    def __init__(self):
        """Adversarial function on output probabilities.
        """
        super(CrossEntropyAdvLoss, self).__init__()

    def forward(self, logits, targets):
        """Adversarial loss function using cross entropy.

        Args:
            logits (torch.FloatTensor): output logits from network, [B, K]
            targets (torch.LongTensor): attack target class
        """
        loss = F.cross_entropy(logits, targets)
        return loss
def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
