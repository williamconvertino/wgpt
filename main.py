import util.cache # Initializes cache in the data directory, to avoid home directory issues on cloud environments

import argparse
from util.config_utils import load_config, load_model_from_config
from data.tokenizer import Tokenizer
from core.trainer import Trainer        

def main():
    
    parser = argparse.ArgumentParser(description="LLM Pretraining Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument("--train", type=str, help="Start training using the specified config name (e.g., gpt2)")
    group.add_argument("--eval", type=str, help="Evaluate using the specified config name (e.g., gpt2)")
    group.add_argument("--gpus", type=int, help="Number of GPUs to use for training", default=2)
    args = parser.parse_args()

    config = load_config(args.train if args.train else args.eval)
    model = load_model_from_config(config)

    if args.train:
        tokenizer = Tokenizer()
        trainer = Trainer(model, tokenizer)
        trainer.train()
    elif args.eval:
        print("Evaluation is not implemented yet.")

if __name__ == "__main__":
    main()