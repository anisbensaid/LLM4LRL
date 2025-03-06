import argparse
import logging
from pathlib import Path
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import evaluate
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path: str, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """Load the fine-tuned model and tokenizer"""
        logger.info(f"Loading model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    def evaluate_perplexity(self, dataset):
        """Calculate perplexity on the test set"""
        logger.info("Calculating perplexity...")
        self.model.eval()
        
        total_loss = 0
        total_length = 0
        
        with torch.no_grad():
            for batch in tqdm(dataset, desc="Evaluating"):
                inputs = self.tokenizer(
                    batch["text"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config["data"]["max_length"]
                ).to(self.device)
                
                outputs = self.model(**inputs)
                loss = outputs.loss
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_length += inputs["input_ids"].size(1)
        
        perplexity = torch.exp(torch.tensor(total_loss / total_length))
        return perplexity.item()

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to fine-tuned model")
    parser.add_argument("--test_data", type=str, required=True,
                      help="Path to test dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                      help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, config)
    
    # Load test dataset
    test_dataset = load_from_disk(args.test_data)
    
    # Calculate perplexity
    perplexity = evaluator.evaluate_perplexity(test_dataset)
    logger.info(f"Test Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    main() 