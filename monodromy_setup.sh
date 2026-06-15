# Install dependencies
pip install torch transformers numpy

# Option A: use the simple create_data.py (small corpus, good for testing)
python create_data.py

# Option B: use WikiText-2 for real results (recommended)
pip install datasets
python -c "
from datasets import load_dataset
import json, re, collections
ds = load_dataset('wikitext', 'wikitext-2-raw-v1')
text = ' '.join(ds['train']['text'])
words = re.findall(r'[a-z]+', text.lower())
freq = collections.Counter(words)
vocab = ['<pad>','<unk>','<eos>'] + [w for w,_ in freq.most_common(1021)]
w2i = {w:i for i,w in enumerate(vocab)}
ids = [w2i.get(w,1) for w in words]
n = int(0.9*len(ids))
import json
with open('/tmp/train_ids.json','w') as f: json.dump(ids[:n],f)
with open('/tmp/val_ids.json','w')   as f: json.dump(ids[n:],f)
with open('/tmp/vocab.json','w')     as f: json.dump(vocab,f)
print(f'Train: {n:,}  Val: {len(ids)-n:,}')
"
