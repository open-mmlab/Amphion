import torch
import torch.nn as nn
import torch.nn.functional as F

cosine_loss = nn.CosineEmbeddingLoss()

class AMSoftmaxLoss(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.3, scale=30):
        super(AMSoftmaxLoss, self).__init__()
        self.m = margin
        self.s = scale
        self.in_feats = nOut # input feature size
        self.nClasses = nClasses # number of classes (speaker numbers)
        self.W = nn.Parameter(torch.randn(nOut, nClasses), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        print('Initialised AMSoftmax m=%.3f s=%.3f'%(self.m,self.s))

    def forward(self, x, label=None):
        # x : B, H
        # label: B
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        label = label.reshape(-1)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda: 
            label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda:
            delt_costh = delt_costh.to(x.device)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, label)
        return loss
    

def noise_clean_similarity_loss(ref_emb, noisy_ref_emb):
    target = torch.tensor([1], dtype=torch.float).to(ref_emb.device)
    ref_emb = ref_emb.reshape(ref_emb.size(0), -1) # (B, 32, 512) --> (B, 32 * 512)
    noisy_ref_emb = noisy_ref_emb.reshape(noisy_ref_emb.size(0), -1) # (B, 32, 512) --> (B, 32 * 512)
    loss = cosine_loss(ref_emb, noisy_ref_emb, target) + 1e-6
    return loss

def cross_entropy_loss(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
class ConstractiveSpeakerLoss(nn.Module):
    def __init__(self, temperature=1.):
        super(ConstractiveSpeakerLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x, speaker_ids):
        # x : B, H
        # speaker_ids: B 3 4 3
        speaker_ids = speaker_ids.reshape(-1)
        speaker_ids_expand = torch.zeros(len(speaker_ids),len(speaker_ids)).to(speaker_ids.device)
        speaker_ids_expand = (speaker_ids.view(-1,1) == speaker_ids).float() #形成一个mask
        x_t = x.transpose(0,1) # B, C --> C,B
        logits = (x @ x_t) / self.temperature # B, H * H, B --> B, B
        targets = F.softmax(speaker_ids_expand / self.temperature, dim=-1)
        loss = cross_entropy_loss(logits, targets, reduction='none')
        return loss.mean()
    
def diff_loss(pred, target, mask, loss_type="l1"):
    # pred: (B, T, d)
    # target: (B, T, d)
    # mask: (B, T)
    if loss_type == "l1":
        loss = F.l1_loss(pred, target, reduction="none").float() * (
            mask.to(pred.dtype).unsqueeze(-1)
        )
    elif loss_type == "l2":
        loss = F.mse_loss(pred, target, reduction="none").float() * (
            mask.to(pred.dtype).unsqueeze(-1)
        )
    else:
        raise NotImplementedError()
    loss = (torch.mean(loss, dim=-1)).sum() / (mask.to(pred.dtype).sum())
    return loss