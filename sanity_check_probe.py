import json
import numpy as np
import random

with open('probe_analysis_results.json', 'r') as f:
    data = json.load(f)

layer_targets = {}
for example in data:
    targets = example['example']['running_counts']
    for layer_idx, layer_data in enumerate(example['layer_results']):
        if layer_idx not in layer_targets:
            layer_targets[layer_idx] = []
        layer_targets[layer_idx].extend(targets) 

from sklearn.metrics import r2_score

print('Sanity check: R² scores after shuffling targets')
for layer_idx in sorted(layer_targets.keys()):
    orig_targets = np.array(layer_targets[layer_idx])
    shuffled_targets = orig_targets.copy()
    np.random.shuffle(shuffled_targets)
    r2 = r2_score(orig_targets, shuffled_targets)
    print(f'Layer {layer_idx}: R² (shuffled) = {r2:.4f}') 