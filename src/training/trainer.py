import argparse
import logging
from pathlib import Path
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model"]["base_model"]
        self.setup_model_and_tokenizer()

    def setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Setup LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            **self.config["model"]["peft_config"]
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(self, dataset_path: str):
        """Prepare dataset for training"""
        dataset = load_from_disk(dataset_path)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config["data"]["max_length"],
                padding="max_length",
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=dataset.column_names,
        )

        return tokenized_dataset

    def train(self, train_dataset, eval_dataset):
        """Train the model"""
        training_args = TrainingArguments(
            **self.config["training"],
            report_to="wandb",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
        )

        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        output_dir = Path(self.config["training"]["output_dir"])
        trainer.save_model(output_dir / "final_model")
        logger.info(f"Model saved to {output_dir / 'final_model'}")

def main():
    parser = argparse.ArgumentParser(description="Train LLM with LoRA")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--dataset", type=str, required=True,
                      help="Path to processed dataset")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb
    wandb.init(project="low-resource-llm", config=config)
    
    # Initialize trainer
    trainer = LLMTrainer(config)
    
    # Prepare dataset
    dataset = trainer.prepare_dataset(args.dataset)
    train_test = dataset.train_test_split(
        test_size=config["data"]["train_test_split"]
    )
    
    # Start training
    trainer.train(train_test["train"], train_test["test"])

if __name__ == "__main__":
    main() 