import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datasets import load_dataset, concatenate_datasets

import torch
from transformers import T5Tokenizer, BartTokenizer, HfArgumentParser, AutoTokenizer


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "name for cached train dataset"},
    )
    valid_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "name for cached valid dataset"},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )

class DataProcessor:
    def __init__(self, tokenizer, model_type="t5", max_source_length=512, max_target_length=32):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_type = model_type
        self.hl_token = "<hl>"
        self.sep_token = "<sep>"

  
    def process(self, dataset):
        dataset = dataset.map(self._add_eos_examples)
        dataset = dataset.map(self._add_special_tokens)
        dataset = dataset.map(self._convert_to_features, batched=True)
        
        return dataset
  
    def _add_eos_examples(self, example):
        example['source_text'] = example['source_text'] + " </s>"
        example['target_text'] = example['target_text'] + " </s>"
        return example
  
    def _add_special_tokens(self, example):
        example['source_text'] = example['source_text'].replace("{hl_token}", self.hl_token)    
        example['target_text'] = example['target_text'].replace("{sep_token}", self.sep_token)
        return example
  
    # tokenize the examples
    def _convert_to_features(self, example_batch):
        source_encoding = self.tokenizer.batch_encode_plus(
            example_batch['source_text'],
            max_length=self.max_source_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True, 
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            example_batch['target_text'],
            max_length=self.max_target_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True, 
        )

        encodings = {
            'source_ids': source_encoding['input_ids'], 
            'target_ids': target_encoding['input_ids'],
            'attention_mask': source_encoding['attention_mask'],
        }

        return encodings

def create_qa(dataset):
    dataset['context'] = "question: " + dataset['question'] + "  context: " + dataset['context']
    dataset['answers'] = dataset['answers']['text'][0]

    return dataset

def create_qg(dataset):
    dataset['context'] = "generate question: " + dataset['context']

    return dataset

def main():
    parser = HfArgumentParser((DataTrainingArguments,))

    data_args = parser.parse_args_into_dataclasses()[0]
    model_type = "t5"

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-base")
    tokenizer.add_tokens(['<sep>', '<hl>'])

    # KLUE DATASET
    klue_dataset = load_dataset("klue", "mrc").remove_columns(['guid', 'is_impossible', 'news_category', 'question_type', 'source', 'title'])
    ## QA
    klue_dataset_qa = klue_dataset.map(create_qa)
    klue_dataset_qa = klue_dataset_qa.rename_column('context', 'source_text')
    klue_dataset_qa = klue_dataset_qa.rename_column('answers', 'target_text')
    klue_dataset_qa = klue_dataset_qa.remove_columns('question')
    ## QG
    klue_dataset_qg = klue_dataset.map(create_qg)
    klue_dataset_qg = klue_dataset_qg.rename_column('context', 'source_text')
    klue_dataset_qg = klue_dataset_qg.rename_column('question', 'target_text')
    klue_dataset_qg = klue_dataset_qg.remove_columns('answers')
    
    train_dataset = concatenate_datasets([klue_dataset_qa['train'], klue_dataset_qg['train']]).shuffle()
    valid_dataset = concatenate_datasets([klue_dataset_qa['validation'], klue_dataset_qg['validation']])

    processor = DataProcessor(
        tokenizer,
        model_type=model_type,
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length
    )

    train_dataset = processor.process(train_dataset)
    valid_dataset = processor.process(valid_dataset)

    columns = ["source_ids", "target_ids", "attention_mask"]
    train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)

    if data_args.train_file_name is None:
        train_file_name = f"train_data_qaqg_{model_type}.pt"
        train_path = os.path.join("data", train_file_name)

        valid_file_name = f"valid_data_qaqg_{model_type}.pt"
        valid_path = os.path.join("data", valid_file_name)
    else:
        train_path = os.path.join("data", data_args.train_file_name)
        valid_path = os.path.join("data", data_args.valid_file_name)
    
    torch.save(train_dataset, train_path)
    logger.info(f"saved train dataset at {train_path}")
    
    torch.save(valid_dataset, valid_path)
    logger.info(f"saved validation dataset at {valid_path}")
    
    tokenizer_path = f"{model_type}_qaqg_tokenizer"
    if not os.path.exists(tokenizer_path):
        os.mkdir(tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"saved tokenizer at {tokenizer_path}")


if __name__ == "__main__":
    main()
