import datasets

db = datasets.load_dataset('vivym/midjourney-prompts', cache_dir='./tmp')

captions = list(set(db['train']['prompt']))

print(len(captions))

dataset = datasets.Dataset.from_dict({"text": captions})
dataset.push_to_hub("Geonmo/midjourney-prompts-only")
