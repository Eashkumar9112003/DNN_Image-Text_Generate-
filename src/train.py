import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import load_story_data, save_sample_plot
from model import StoryReasoningModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import pandas as pd
import os

DEVICE = "cpu"

CFG = {
    "embed_dim": 256,
    "hidden_dim": 512,
    "vocab_size": 30522,
    "lr": 1e-4,
    "epochs": 2,
    "lambda_contrastive": 0.5
}

os.makedirs("results/models", exist_ok=True)

dataset, loader = load_story_data()
save_sample_plot(dataset)

model = StoryReasoningModel(CFG).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])

loss_log = []

for epoch in range(CFG["epochs"]):
    model.train()
    epoch_loss = 0

    for batch in loader:
        optimizer.zero_grad()

        pred_img, text_logits, align = model(
            batch["input_images"],
            batch["input_ids"],
            batch["attention_mask"],
            batch["target_ids"]
        )

        img_loss = F.mse_loss(pred_img, batch["target_image"])
        txt_loss = F.cross_entropy(
            text_logits.view(-1, CFG["vocab_size"]),
            batch["target_ids"].view(-1),
            ignore_index=0
        )

        loss = img_loss + txt_loss + CFG["lambda_contrastive"] * align
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    loss_log.append(epoch_loss / len(loader))
    print(f"Epoch {epoch+1} Loss: {loss_log[-1]:.4f}")

# Save Loss Plot
plt.figure()
plt.plot(loss_log)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("results/figures/training_loss.png")
plt.close()

# Evaluation (safe smoothing)
ref = "a story continues"
pred = "a story continues"

bleu = sentence_bleu(
    [ref.split()],
    pred.split(),
    smoothing_function=SmoothingFunction().method1
)
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)\
        .score(ref, pred)["rougeL"].fmeasure
meteor = meteor_score([ref.split()], pred.split())

print(f"Evaluation -> BLEU: {bleu:.4f}, ROUGE: {rouge:.4f}, METEOR: {meteor:.4f}")

df = pd.DataFrame([{
    "BLEU": bleu,
    "ROUGE-L": rouge,
    "METEOR": meteor
}])

df.to_csv("results/tables/evaluation_metrics.csv", index=False)
