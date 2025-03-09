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

# Change logging level to DEBUG to see more information
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LLMTrainer:
    def __init__(self, config: dict):
        logger.debug("Initializing LLMTrainer")
        self.config = config
        self.model_name = config["model"]["base_model"]
        logger.debug(f"Using model: {self.model_name}")
        self.setup_model_and_tokenizer()

    def setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer"""
        logger.info("Setting up tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer loaded successfully")
            
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Model loaded successfully")
            
            logger.info("Setting up LoRA...")
            peft_config_dict = self.config["model"]["peft_config"].copy()
            if "task_type" in peft_config_dict:
                del peft_config_dict["task_type"]
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                **peft_config_dict
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            logger.info("LoRA setup complete")
            
        except Exception as e:
            logger.error(f"Error in setup_model_and_tokenizer: {str(e)}", exc_info=True)
            raise

    def prepare_dataset(self, dataset_path: str):
        """Prepare dataset for training"""
        logger.info(f"Loading dataset from {dataset_path}")
        try:
            dataset = load_from_disk(dataset_path)
            logger.info(f"Dataset loaded with {len(dataset)} examples")
            
            logger.info("Tokenizing dataset...")
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
                desc="Tokenizing dataset"
            )
            logger.info("Dataset tokenization complete")
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"Error in prepare_dataset: {str(e)}", exc_info=True)
            raise

    def train(self, train_dataset, eval_dataset):
        """Train the model"""
        logger.info("Setting up training arguments")
        try:
            training_args = TrainingArguments(
                **self.config["training"],
                report_to="wandb",
            )
            
            logger.info("Creating trainer instance")
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=default_data_collator,
            )

            logger.info("Starting training...")
            trainer.train()
            
            logger.info("Training completed, saving model...")
            output_dir = Path(self.config["training"]["output_dir"])
            trainer.save_model(output_dir / "final_model")
            logger.info(f"Model saved to {output_dir / 'final_model'}")
            
        except Exception as e:
            logger.error(f"Error in train: {str(e)}", exc_info=True)
            raise

def main():
    try:
        logger.info("Starting trainer script")
        parser = argparse.ArgumentParser(description="Train LLM with LoRA")
        parser.add_argument("--config", type=str, default="configs/config.yaml",
                          help="Path to configuration file")
        parser.add_argument("--dataset", type=str, required=True,
                          help="Path to processed dataset")
        
        args = parser.parse_args()
        logger.debug(f"Arguments: {args}")
        
        logger.info(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.debug(f"Loaded config: {config}")
        
        logger.info("Initializing wandb")
        wandb.init(project="low-resource-llm", config=config)
        
        logger.info("Creating trainer instance")
        trainer = LLMTrainer(config)
        
        logger.info(f"Preparing dataset from {args.dataset}")
        dataset = trainer.prepare_dataset(args.dataset)
        
        logger.info("Splitting dataset")
        train_test = dataset.train_test_split(
            test_size=config["data"]["train_test_split"]
        )
        
        logger.info("Starting training process")
        trainer.train(train_test["train"], train_test["test"])
        
    except Exception as e:
        logger.error(f"Main function error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 