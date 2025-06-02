import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
with open('probe_analysis_results.json', 'r') as f:
    data = json.load(f)

# Extract layer-wise metrics
layer_metrics = {}
for example in data:
    for layer_idx, layer_data in enumerate(example['layer_results']):
        if layer_idx not in layer_metrics:
            layer_metrics[layer_idx] = {
                'r2_scores': [],
                'coefficients': []
            }
        layer_metrics[layer_idx]['r2_scores'].append(layer_data['r2_score'])
        layer_metrics[layer_idx]['coefficients'].append(layer_data['probe_coefficients'])

# Calculate average R² scores per layer
avg_r2_scores = [np.mean(metrics['r2_scores']) for layer_idx, metrics in sorted(layer_metrics.items())]

# Plot R² scores
plt.figure(figsize=(12, 6))
plt.plot(avg_r2_scores, marker='o')
plt.title('Average R² Scores by Layer')
plt.xlabel('Layer')
plt.ylabel('Average R² Score')
plt.grid(True)
plt.savefig('avg_r2_scores.png')

# Print layer with highest R² score
best_layer = np.argmax(avg_r2_scores)
print(f"Layer with highest average R² score: {best_layer}")
print(f"Highest average R² score: {max(avg_r2_scores):.4f}")

# Analyze coefficient patterns
coefficient_patterns = []
for layer_idx, metrics in sorted(layer_metrics.items()):
    avg_coefficients = np.mean(metrics['coefficients'], axis=0)
    coefficient_patterns.append(avg_coefficients)

# Plot coefficient heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(coefficient_patterns, cmap='RdBu_r', center=0)
plt.title('Average Probe Coefficients by Layer')
plt.xlabel('Feature Dimension')
plt.ylabel('Layer')
plt.savefig('coefficient_patterns.png')

# Print summary statistics
print("\nSummary Statistics:")
for layer_idx, metrics in sorted(layer_metrics.items()):
    r2_scores = metrics['r2_scores']
    print(f"\nLayer {layer_idx}:")
    print(f"  Average R² score: {np.mean(r2_scores):.4f}")
    print(f"  Std R² score: {np.std(r2_scores):.4f}")
    print(f"  Max R² score: {np.max(r2_scores):.4f}")
    print(f"  Min R² score: {np.min(r2_scores):.4f}") 