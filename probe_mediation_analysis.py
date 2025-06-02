import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import ast

class ProbeMediationAnalyzer:
    def __init__(self, benchmark_data_path, answers_path):
        """Initialize the analyzer with benchmark data and model."""
        self.benchmark_data = pd.read_csv(benchmark_data_path)
        self.answers_data = pd.read_csv(answers_path)
        
        # Load Pythia model
        print("Loading Pythia model...")
        self.model_name = "Nexusflow/Starling-LM-7B-beta"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            output_hidden_states=True,
            torch_dtype=torch.float32
        )
        print("Model loaded successfully!")

    def get_hidden_states(self, prompt):
        """Get hidden states from the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.hidden_states

    def create_prompt(self, words, target_word):
        """Create a prompt for counting target word occurrences."""
        return f"Count how many times the word '{target_word}' appears in this list: {', '.join(words)}. The count is:"

    def get_intermediate_prompts(self, words, target_word):
        """Create prompts for each position in the sequence."""
        prompts = []
        for i in range(1, len(words) + 1):
            current_words = words[:i]
            prompt = self.create_prompt(current_words, target_word)
            prompts.append(prompt)
        return prompts

    def calculate_running_counts(self, words, target_word):
        """Calculate running counts of target word."""
        running_counts = []
        count = 0
        for word in words:
            if word.strip() == target_word:
                count += 1
            running_counts.append(count)
        return running_counts

    def train_probe(self, hidden_states, running_counts):
        """Train a linear probe to predict running counts from hidden states."""
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(hidden_states, running_counts, test_size=0.2, random_state=42)
        
        probe = LinearRegression()
        probe.fit(X_train, y_train)
        predictions = probe.predict(X_test)
        r2 = r2_score(y_test, predictions)
        return probe, r2

    def analyze_layer_encoding(self):
        """Analyze which layers encode the running count information."""
        results = []
        
        # Select examples for analysis
        examples = self.benchmark_data.sample(min(10, len(self.benchmark_data)))
        
        for _, row in tqdm(examples.iterrows(), total=len(examples), desc="Analyzing examples"):
            words = ast.literal_eval(row['list'])
            target_word = row['type']
            
            # Get running counts
            running_counts = self.calculate_running_counts(words, target_word)
            
            # Get intermediate prompts
            prompts = self.get_intermediate_prompts(words, target_word)
            
            # Get hidden states for each prompt
            all_hidden_states = []
            for prompt in prompts:
                hidden_states = self.get_hidden_states(prompt)
                all_hidden_states.append(hidden_states)
            
            # For each layer, train a probe and evaluate
            layer_results = []
            for layer_idx in range(len(all_hidden_states[0])):
                # Extract hidden states for this layer across all positions
                layer_states = []
                for hidden_states in all_hidden_states:
                    # Take the average of the last 3 tokens' hidden states
                    state = hidden_states[layer_idx][0, -3:].mean(dim=0).numpy()
                    layer_states.append(state)
                layer_states = np.array(layer_states)
                
                # Train probe
                probe, r2 = self.train_probe(layer_states, running_counts)
                
                layer_results.append({
                    'layer': layer_idx,
                    'r2_score': float(r2),
                    'probe_coefficients': probe.coef_.tolist(),
                    'probe_intercept': float(probe.intercept_)
                })
            
            results.append({
                'example': {
                    'words': words,
                    'target_word': target_word,
                    'running_counts': running_counts
                },
                'layer_results': layer_results
            })
        
        # Save results
        with open('probe_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualization
        self.visualize_results(results)

    def visualize_results(self, results):
        """Create visualizations of the probe analysis results."""
        # Average R² scores per layer
        avg_r2_scores = []
        for layer_idx in range(len(results[0]['layer_results'])):
            layer_r2_scores = [r['layer_results'][layer_idx]['r2_score'] for r in results]
            avg_r2_scores.append(np.mean(layer_r2_scores))
        
        # Plot average R² scores
        plt.figure(figsize=(12, 6))
        plt.plot(avg_r2_scores, marker='o')
        plt.title('Average R² Score by Layer')
        plt.xlabel('Layer')
        plt.ylabel('R² Score')
        plt.grid(True)
        plt.savefig('layer_r2_scores.png')
        plt.close()
        
        # Heatmap of probe coefficients
        best_layer_idx = np.argmax(avg_r2_scores)
        best_layer_coefficients = np.array([r['layer_results'][best_layer_idx]['probe_coefficients'] 
                                          for r in results])
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(best_layer_coefficients, cmap='RdBu_r', center=0)
        plt.title(f'Probe Coefficients for Best Layer (Layer {best_layer_idx})')
        plt.xlabel('Hidden Unit')
        plt.ylabel('Example')
        plt.savefig('probe_coefficients_heatmap.png')
        plt.close()

    def perform_causal_mediation(self, source_words, target_words, target_word, layer_idx):
        """Perform causal mediation analysis using the identified layer."""
        # Get hidden states for source and target sequences
        source_prompts = self.get_intermediate_prompts(source_words, target_word)
        target_prompts = self.get_intermediate_prompts(target_words, target_word)
        
        source_states = [self.get_hidden_states(p) for p in source_prompts]
        target_states = [self.get_hidden_states(p) for p in target_prompts]
        
        # Perform patching experiment
        patched_states = []
        for src_state, tgt_state in zip(source_states, target_states):
            # Create a copy of target states
            patched = [h.clone() for h in tgt_state]
            # Patch the identified layer
            patched[layer_idx] = src_state[layer_idx]
            patched_states.append(patched)
        
        return patched_states

if __name__ == "__main__":
    analyzer = ProbeMediationAnalyzer(
        'count_matching_words_benchmark.csv',
        'count_matching_words_benchmark_with_answers.csv'
    )
    analyzer.analyze_layer_encoding() 