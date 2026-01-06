import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import DistilBertModel

class VisualEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        resnet = models.resnet18(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, embed_dim)

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b*s, c, h, w)
        f = self.backbone(x).view(b*s, -1)
        f = self.fc(f)
        return f.view(b, s, -1)

class TextEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.fc = nn.Linear(768, embed_dim)

    def forward(self, ids, mask):
        b, s, l = ids.shape
        ids = ids.view(b*s, l)
        mask = mask.view(b*s, l)
        out = self.bert(ids, attention_mask=mask)
        cls = out.last_hidden_state[:,0]
        return self.fc(cls).view(b, s, -1)

def contrastive_loss(t, v, temp=0.07):
    t = F.normalize(t, dim=1)
    v = F.normalize(v, dim=1)
    sim = torch.matmul(t, v.T) / temp
    labels = torch.arange(sim.size(0), device=sim.device)
    return (F.cross_entropy(sim, labels) +
            F.cross_entropy(sim.T, labels)) / 2

class StoryReasoningModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.visual = VisualEncoder(cfg["embed_dim"])
        self.text   = TextEncoder(cfg["embed_dim"])

        self.fusion = nn.Linear(cfg["embed_dim"]*2, cfg["hidden_dim"])
        self.lstm   = nn.LSTM(cfg["hidden_dim"], cfg["hidden_dim"], batch_first=True)

        self.text_emb = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.text_dec = nn.GRU(cfg["embed_dim"], cfg["hidden_dim"], batch_first=True)
        self.text_out = nn.Linear(cfg["hidden_dim"], cfg["vocab_size"])

        self.img_dec = nn.Sequential(
            nn.Linear(cfg["hidden_dim"], 256*7*7),
            nn.Unflatten(1, (256,7,7)),
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,3,4,2,1),
            nn.Sigmoid()
        )

    def forward(self, imgs, ids, mask, tgt_ids=None):
        v = self.visual(imgs)
        t = self.text(ids, mask)

        B,S,E = v.shape
        align = contrastive_loss(
            t.reshape(B*S,E),
            v.reshape(B*S,E)
        )

        fused = torch.relu(self.fusion(torch.cat([v,t],dim=-1)))
        _, (h,_) = self.lstm(fused)
        ctx = h[-1]

        pred_img = self.img_dec(ctx)

        text_logits = None
        if tgt_ids is not None:
            emb = self.text_emb(tgt_ids)
            out,_ = self.text_dec(emb, ctx.unsqueeze(0))
            text_logits = self.text_out(out)

        return pred_img, text_logits, align
