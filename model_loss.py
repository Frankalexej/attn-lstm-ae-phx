import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLoss: 
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def get_loss(self, y_hat, y, mask): 
        """
        Computes the masked loss given a loss function, predicted values, ground truth values, and a mask tensor.
        """
        b, t, f = y_hat.shape
        mask = mask.unsqueeze(-1).expand((b, t, f)).bool()  # both y_hat and y are of same size
        # Apply mask to the predicted and ground truth values
        y_hat_masked = y_hat.masked_select(mask)
        y_masked = y.masked_select(mask)

        # Calculate the loss using the masked values
        loss = torch.sum(self.loss_fn(y_hat_masked, y_masked)) / torch.sum(mask)
        # loss = loss_fn(y_hat_masked, y_masked)
        return loss
    
class AlphaCombineLoss:
    def __init__(self, recon_loss, pred_loss, alpha=0.1):
        self.recon_loss = recon_loss
        self.pred_loss = pred_loss
        self.alpha = alpha
    
    def get_loss(self, y_hat_recon, y_recon, y_hat_pred, y_pred, x_lens, y_pred_lens, mask): 
        reconstruction_loss = self.recon_loss.get_loss(y_hat_recon, y_recon, mask)
        prediction_loss = self.pred_loss(y_hat_pred, y_pred, x_lens, y_pred_lens)

        # Compute the regularization term based on the number of boundaries
        prediction_loss_term = self.alpha * prediction_loss

        return reconstruction_loss + prediction_loss_term, (reconstruction_loss, prediction_loss)

class PseudoAlphaCombineLoss_Recon:
    def __init__(self, recon_loss, pred_loss, alpha=0.1):
        self.recon_loss = recon_loss
        self.pred_loss = pred_loss
        self.alpha = alpha
    
    def get_loss(self, y_hat_recon, y_recon, y_hat_pred, y_pred, x_lens, y_pred_lens, mask): 
        reconstruction_loss = self.recon_loss.get_loss(y_hat_recon, y_recon, mask)

        return reconstruction_loss, (reconstruction_loss, torch.tensor(0))

    
class PseudoAlphaCombineLoss_Pred:
    def __init__(self, recon_loss, pred_loss, alpha=0.1):
        self.recon_loss = recon_loss
        self.pred_loss = pred_loss
        self.alpha = alpha
    
    def get_loss(self, y_hat_recon, y_recon, y_hat_pred, y_pred, x_lens, y_pred_lens, mask): 
        prediction_loss = self.pred_loss(y_hat_pred, y_pred, x_lens, y_pred_lens)

        return prediction_loss, (torch.tensor(0), prediction_loss)
    
class MaskedFlatLoss: 
    # This used for CrossEntropyLoss, which needs (B, C, L) and (B, L) as input and will give
    # (B, L) as output. Therefore masking can simply be appied with mask being of shape (B, L)
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def get_loss(self, y_hat, y, mask): 
        y_hat = y_hat.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
        loss = torch.sum(self.loss_fn(y_hat, y) * mask) / torch.sum(mask)
        return loss


class CombinedLoss: 
    def __init__(self, loss_class, alpha=0.1):
        self.loss_class = loss_class
        self.alpha = alpha
    
    def get_loss(self, y_hat, y, mask, z2): 
        # h2 is second layer output of encoder HMLSTM
        # of shape (B, L)
        reconstruction_loss = self.loss_class.get_loss(y_hat, y, mask)

        b, t, f = z2.shape
        mask = mask.unsqueeze(-1).expand((b, t, f)).bool()  # both y_hat and y are of same size
        # Apply mask to the predicted and ground truth values
        z2_masked = z2.masked_select(mask)

        # Calculate the number of boundaries for each sequence in the batch
        reg_loss = torch.sum(z2_masked) / torch.sum(mask)

        # Compute the regularization term based on the number of boundaries
        regularization_term = self.alpha * reg_loss

        return (reconstruction_loss, regularization_term)
    

class MaskedKLDivLoss: 
    def __init__(self):
        pass
    def get_loss(self, mu, logvar, mask): 
        """
        mask: (batch, len)
        mu, logvar: (batch, len, hiddim)
        """
        de_feature_loss = -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp()), 
                                          dim=-1)
        loss = torch.sum(de_feature_loss * mask) / torch.sum(mask)
        return loss