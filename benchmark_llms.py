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

# Load environment variables
load_dotenv()

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class LLMBenchmark:
    def __init__(self):
        self.benchmark_data = pd.read_csv('count_matching_words_benchmark.csv')
        self.answers_data = pd.read_csv('count_matching_words_benchmark_with_answers.csv')
        
        # Initialize Ollama models
        self.ollama_models = [
            # "mistral",
            # "llama2",
            # "codellama",
            # "neural-chat",
            # "starling-lm"
        ]
        
        # Initialize Together API models
        self.together_models = [
            # "mistralai/Mistral-7B-Instruct-v0.2",
            # "meta-llama/Llama-2-7b-chat-hf",
            # "tiiuae/falcon-7b-instruct"
        ]
        
        # Initialize HuggingFace models
        self.hf_models = [
            # "facebook/opt-1.3b",
            # "EleutherAI/pythia-1.4b",
            # "bigscience/bloom-560m",
            # "Qwen/Qwen3-0.6B",
            # "Qwen/Qwen3-1.7B",
            "Nexusflow/Starling-LM-7B-beta",
        ]
        
        # Set up Together API
        together.api_key = os.getenv("TOGETHER_API_KEY")
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
            if "Qwen3" in model_name:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,  # Use float16 for better memory efficiency
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    offload_folder="offload"  # Specify offload folder
                )
                
                # Format the prompt as a chat message for Qwen3
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False  # Disable thinking mode for counting task
                )
                inputs = tokenizer([text], return_tensors="pt").to(model.device)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate response with more conservative settings
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,  # Reduced from 100
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Extract the response (excluding the input prompt)
            output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
            response = tokenizer.decode(output_ids, skip_special_tokens=True)
            
            # Clear CUDA cache after each generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return response
            
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
        correct = 0
        total = len(self.benchmark_data)
        # total = 10
        results = []

        for idx, row in tqdm(self.benchmark_data.iterrows(), total=total, desc=f"Evaluating {model_name}"):
            if idx > total:
                break
            prompt = self.create_prompt(row)
            ground_truth = int(self.answers_data.iloc[idx]['answer'])  # Convert to Python int
            
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
                "prediction": int(prediction),  # Convert to Python int
                "ground_truth": ground_truth,
                "is_correct": bool(is_correct)  # Convert to Python bool
            })

        accuracy = float(correct / total)  # Convert to Python float
        return {
            "model_name": model_name,
            "model_type": model_type,
            "accuracy": accuracy,
            "correct": int(correct),  # Convert to Python int
            "total": int(total),  # Convert to Python int
            "results": results
        }

    def run_benchmark(self):
        """Run the benchmark on all models."""
        results = []
        
        # Run Ollama models
        for model in self.ollama_models:
            result = self.evaluate_model(model, "ollama")
            results.append(result)
            
        # Run Together API models
        for model in self.together_models:
            result = self.evaluate_model(model, "together")
            results.append(result)
            
        # Run HuggingFace models
        for model in self.hf_models:
            result = self.evaluate_model(model, "huggingface")
            results.append(result)
        
        # Save results using custom encoder
        with open('benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        # Print summary
        print("\nBenchmark Results Summary:")
        print("-" * 50)
        for result in results:
            print(f"{result['model_name']} ({result['model_type']}):")
            print(f"Accuracy: {result['accuracy']:.2%}")
            print(f"Correct: {result['correct']}/{result['total']}")
            print("-" * 50)

if __name__ == "__main__":
    benchmark = LLMBenchmark()
    benchmark.run_benchmark() 