import pandas as pd
from datasets import Dataset


# Download "Train_GCC-training.tsv" from https://ai.google.com/research/ConceptualCaptions/download
csv = pd.read_csv("Train_GCC-training.tsv", sep="\t", names=["caption", "img_url"])
captions = csv["caption"]

dataset = Dataset.from_dict({"text": captions})
dataset.save_to_disk("gcc_caption_only")