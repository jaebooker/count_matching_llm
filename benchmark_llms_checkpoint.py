import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import together

load_dotenv()

class LLMBenchmark:
    def __init__(self):
        self.benchmark_data = pd.read_csv('count_matching_words_benchmark.csv')
        self.answers_data = pd.read_csv('count_matching_words_benchmark_with_answers.csv')
        
        self.ollama_models = [
            # "mistral",
            # "llama2",
            "codellama",
            # "neural-chat",
            "starling-lm"
        ]
        
        self.together_models = [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b-chat-hf",
            "tiiuae/falcon-7b-instruct"
        ]
        
        self.hf_models = [
            "facebook/opt-1.3b",
            "EleutherAI/pythia-1.4b",
            "bigscience/bloom-560m"
        ]
        
        together.api_key = ["YOUR_API_KEY"]
        
        self.results = self.load_checkpoint()
        
    def load_checkpoint(self) -> List[Dict[str, Any]]:
        try:
            if os.path.exists('benchmark_results_checkpoint.json'):
                with open('benchmark_results_checkpoint.json', 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
        return []
        
    def save_checkpoint(self, result: Dict[str, Any]):
        try:
            self.results.append(result)
            with open('benchmark_results_checkpoint.json', 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\nSaved checkpoint for {result['model_name']}")
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")

    def get_ollama_response(self, model: str, prompt: str) -> str:
        """Get response from Ollama model."""
        try:
            response = requests.post('http://localhost:11434/api/generate',
                                  json={
                                      "model": model,
                                      "prompt": prompt,
                                      "stream": False
                                  })
            return response.json()['response']
        except Exception as e:
            print(f"Error with Ollama model {model}: {str(e)}")
            return ""

    def get_together_response(self, model: str, prompt: str) -> str:
        """Get response from Together API model."""
        try:
            response = together.Complete.create(
                prompt=prompt,
                model=model,
                max_tokens=50,
                temperature=0.1,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1.1,
                stop=['</s>', '\n\n']
            )
            return response['output']['choices'][0]['text']
        except Exception as e:
            print(f"Error with Together model {model}: {str(e)}")
            return ""

    def get_hf_response(self, model_name: str, prompt: str) -> str:
        """Get response from HuggingFace model."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                inputs["input_ids"],
                max_length=100,
                temperature=0.1,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error with HuggingFace model {model_name}: {str(e)}")
            return ""

    def create_prompt(self, row: pd.Series) -> str:
        """Create a prompt for the LLM."""
        return f"""Given a list of words and a category, count how many words in the list belong to that category.
Category: {row['type']}
List: {row['list']}
Count: """

    def extract_number(self, response: str) -> int:
        """Extract a number from the LLM's response."""
        try:
            # Try to find the first number in the response
            words = response.split()
            for word in words:
                if word.isdigit():
                    return int(word)
            return 0
        except:
            return 0

    def evaluate_model(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """Evaluate a model on the benchmark."""
        # Check if this model has already been evaluated
        for result in self.results:
            if result['model_name'] == model_name and result['model_type'] == model_type:
                print(f"\nSkipping {model_name} as it was already evaluated")
                return result

        correct = 0
        total = len(self.benchmark_data)
        results = []

        for idx, row in tqdm(self.benchmark_data.iterrows(), total=total, desc=f"Evaluating {model_name}"):
            prompt = self.create_prompt(row)
            ground_truth = self.answers_data.iloc[idx]['answer']
            
            if model_type == "ollama":
                response = self.get_ollama_response(model_name, prompt)
            elif model_type == "together":
                response = self.get_together_response(model_name, prompt)
            else:  # huggingface
                response = self.get_hf_response(model_name, prompt)
            
            prediction = self.extract_number(response)
            is_correct = prediction == ground_truth
            correct += int(is_correct)
            
            results.append({
                "prompt": prompt,
                "response": response,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "is_correct": is_correct
            })

        accuracy = correct / total
        result = {
            "model_name": model_name,
            "model_type": model_type,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }
        
        # Save checkpoint after each model
        self.save_checkpoint(result)
        
        return result

    def run_benchmark(self):
        """Run the benchmark on all models."""
        # Run Ollama models
        for model in self.ollama_models:
            result = self.evaluate_model(model, "ollama")
            
        # Run Together API models
        for model in self.together_models:
            result = self.evaluate_model(model, "together")
            
        # Run HuggingFace models
        for model in self.hf_models:
            result = self.evaluate_model(model, "huggingface")
        
        # Save final results
        with open('benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        print("\nBenchmark Results Summary:")
        print("-" * 50)
        for result in self.results:
            print(f"{result['model_name']} ({result['model_type']}):")
            print(f"Accuracy: {result['accuracy']:.2%}")
            print(f"Correct: {result['correct']}/{result['total']}")
            print("-" * 50)

if __name__ == "__main__":
    benchmark = LLMBenchmark()
    benchmark.run_benchmark() 