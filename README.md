# LLM Count Matching Benchmark

This benchmark evaluates various open-weight language models on a word counting task. The task involves counting how many words in a given list belong to a specific category.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file in the project root with your API keys:
```
TOGETHER_API_KEY=your_together_api_key_here
```

3. Install Ollama (for local model inference):
Follow the instructions at https://ollama.ai/download to install Ollama on your system.

4. Pull the required Ollama models:
```bash
ollama pull mistral
ollama pull llama2
ollama pull codellama
ollama pull neural-chat
ollama pull starling-lm
```

## Running the Benchmark

Simply run:
```bash
python benchmark_llms.py
```

The script will:
1. Test local models using Ollama
2. Test models through the Together API
3. Test models from HuggingFace
4. Save detailed results to `benchmark_results.json`
5. Print a summary of results to the console

## Models Tested

### Local Models (via Ollama)
- Mistral
- Llama2
- CodeLlama
- Neural Chat
- Starling LM

### Together API Models
- Mistral-7B-Instruct-v0.2
- Llama-2-7b-chat-hf
- Falcon-7B-Instruct

### HuggingFace Models
- OPT-1.3B
- Pythia-1.4B
- BLOOM-560M

## Results

The benchmark results are saved in `benchmark_results.json` with the following information for each model:
- Model name and type
- Overall accuracy
- Number of correct predictions
- Detailed results for each test case

## Notes

- The benchmark uses a temperature of 0.1 to ensure consistent results
- For API-based models, you'll need valid API keys
- Local models require sufficient GPU memory
- The script includes error handling for failed API calls or model loading issues 