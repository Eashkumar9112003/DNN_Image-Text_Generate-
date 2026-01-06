import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import DistilBertTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import nltk

# Safe downloads (no crash)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)

# Directories
FIG_DIR = "results/figures"
TAB_DIR = "results/tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

CONFIG = {
    "batch_size": 2,
    "seq_len": 4,
    "max_text_len": 32,
    "dataset_size": 300
}

class StoryDataset(Dataset):
    def __init__(self, data, tokenizer, transform, seq_len):
        self.data = data
        self.tokenizer = tokenizer
        self.transform = transform
        self.need = seq_len + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        images = item["images"]
        texts = item["story"]

        if isinstance(texts, str):
            texts = [texts]

        images += [images[-1]] * (self.need - len(images))
        texts  += [texts[-1]]  * (self.need - len(texts))

        images = images[-self.need:]
        texts  = texts[-self.need:]

        imgs = torch.stack([
            self.transform(img.convert("RGB")) for img in images
        ])

        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_text_len"],
            return_tensors="pt"
        )

        return {
            "input_images": imgs[:-1],
            "input_ids": enc["input_ids"][:-1],
            "attention_mask": enc["attention_mask"][:-1],
            "target_image": imgs[-1],
            "target_ids": enc["input_ids"][-1]
        }

def load_story_data():
    print("Streaming dataset from Hugging Face (NO DISK WRITE)...")

    stream = load_dataset(
        "daniel3303/StoryReasoning",
        split="train",
        streaming=True
    )

    samples = []
    for i, item in enumerate(stream):
        if i >= CONFIG["dataset_size"]:
            break
        samples.append(item)

    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

    dataset = StoryDataset(samples, tokenizer, transform, CONFIG["seq_len"])
    loader  = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    with open(f"{TAB_DIR}/dataset_info.txt", "w") as f:
        f.write(f"Samples: {len(dataset)}\n")

    return dataset, loader

def save_sample_plot(dataset):
    sample = dataset[0]
    imgs = sample["input_images"]

    plt.figure(figsize=(12, 3))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(imgs[i].permute(1,2,0))
        plt.axis("off")
        plt.title(f"Input {i+1}")

    plt.suptitle("Sample Story Sequence")
    plt.savefig(f"{FIG_DIR}/sample_story.png")
    plt.close()
