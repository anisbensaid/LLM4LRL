import argparse
import logging
from pathlib import Path
import yaml
from datasets import load_from_disk, load_dataset
import re
from typing import List, Dict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self, config: Dict):
        self.config = config
        self.max_length = config["data"]["max_length"]

    def clean_text(self, text: str) -> str:
        """Clean text based on configuration"""
        if self.config["data"]["preprocessing"]["remove_html"]:
            text = re.sub(r'<[^>]+>', '', text)
        
        if self.config["data"]["preprocessing"]["remove_special_chars"]:
            text = re.sub(r'[^\w\s]', '', text)
            
        if self.config["data"]["preprocessing"]["lowercase"]:
            text = text.lower()
            
        return text.strip()

    def process_dataset(self, dataset):
        """Process the dataset according to configuration"""
        logger.info("Processing dataset...")
        
        def process_example(example):
            if "text" in example:
                example["text"] = self.clean_text(example["text"])
            return example

        processed_dataset = dataset.map(
            process_example,
            num_proc=4,
            desc="Cleaning texts"
        )

        return processed_dataset

def main():
    parser = argparse.ArgumentParser(description="Preprocess collected data")
    parser.add_argument("--input", type=str, required=True, help="Input data directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(config)
    
    # Load and process dataset
    if args.input.endswith('.json'):
        dataset = load_dataset('json', data_files=args.input)['train']  # Add ['train'] to get the actual dataset
    else:
        dataset = load_from_disk(args.input)
    processed_dataset = preprocessor.process_dataset(dataset)
    
    # Save processed dataset
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    processed_dataset.save_to_disk(output_path)
    logger.info(f"Processed dataset saved to {output_path}")

if __name__ == "__main__":
    main() 