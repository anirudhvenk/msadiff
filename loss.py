import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alpha = config.loss.kld_loss_scale
        self.beta = config.loss.perm_loss_scale
        self.msa_depth = config.data.msa_depth
        
        self.reconstruction_loss = ReconstructionLoss()
        self.perm_loss = PermutaionMatrixPenalty()
        self.kld_loss = KLDLoss()
        
    def forward(self, msa_true, msa_pred, mask, perm, mu, logvar):
        recon_loss, ppl = self.reconstruction_loss(
            msa_true,
            msa_pred,
            mask,
            self.msa_depth
        )
        perm_loss = self.perm_loss(perm)
        kld_loss = self.kld_loss(mu, logvar)
        total_loss = recon_loss + self.alpha * kld_loss + self.beta * kld_loss
        
        loss_dict = {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "perm_loss": perm_loss,
            "kld_loss": kld_loss,
            "ppl": ppl
        }
        
        return loss_dict

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, msa_true, msa_pred, mask, msa_depth):
        mask_expanded = mask.unsqueeze(1)
        mask_expanded = mask_expanded.expand(-1, msa_depth, -1)
        
        msa_true = msa_true[~mask_expanded]
        msa_pred = msa_pred[~mask_expanded.unsqueeze(-1).expand(-1, -1, -1, 33)].view(-1, 33)

        loss = nn.CrossEntropyLoss(reduction="none")(msa_pred, msa_true)
        perplexity = loss.float().exp().mean()
        loss = loss.mean()
        
        return loss, perplexity

class PermutaionMatrixPenalty(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def entropy(p, axis, normalize=True, eps=10e-12):
        if normalize:
            p = p / (p.sum(axis=axis, keepdim=True) + eps)
        e = - torch.sum(p * torch.clamp_min(torch.log(p), -100), axis=axis)
        return e

    def forward(self, perm, eps=10e-8):
        #print(perm.shape)
        perm = perm + eps
        entropy_col = self.entropy(perm, axis=1, normalize=False)
        entropy_row = self.entropy(perm, axis=2, normalize=False)
        loss = entropy_col.mean() + entropy_row.mean()
        return loss

class KLDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1)
        loss = torch.mean(loss)
        return loss