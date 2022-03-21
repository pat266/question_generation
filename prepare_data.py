import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import nlp
from datasets import load_dataset, concatenate_datasets
from pathlib import Path
from transformers import T5Tokenizer, BartTokenizer, HfArgumentParser

from pprint import pprint


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task: str = field(
        metadata={"help": "Which task 'qa', 'qg', 'e2e_qg', 'ans_ext', 'multi'. 'multi' means 'qa', 'qg', 'ans_ext' tasks"}, 
    )
    model_type: str = field(metadata={"help": "One of 't5', 'bart'"})
    dataset_path: Optional[str] = field(
        default="data/squad_multitask",
        metadata={"help": "Path for dataset directory"}, 
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "name for the json train file"},
    )
    valid_file: Optional[str] = field(
        default=None,
        metadata={"help": "name for the json valid file"},
    )
    train_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "name for cached train dataset"},
    )
    valid_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "name for cached valid dataset"},
    )
    load_local: bool = field(
        default=False,
        metadata={"help": "For loading local json files."}
    )
    valid_for_qg_only: bool = field(
        default=False,
        metadata={"help": "For multitask dataset valid split should contain only qg task or all tasks."}
    )
    qg_format: Optional[str] = field(
        default='highlight_qg_format',
        metadata={"help": "How to format inputs for que generation, 'highlight_qg_format' or 'prepend_qg_format'"}, 
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
        
        if model_type == "t5":
            self.sep_token = "<sep>"
        elif model_type == "bart":
            self.sep_token = "<sep>"
        else:
            self.sep_token = "[SEP]"
  
    def process(self, dataset):
        if self.model_type == "t5":
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


def filter_qa(example):
    return example['task'] == 'qa'

def filter_qg(example):
    return example['task'] == 'qg'

def filter_e2e_qg(example):
    return example['task'] == 'e2e_qg'

def filter_ans_ext(example):
    return example['task'] == 'ans_ext'

def filter_multi(example):
    return example['task'] != 'e2e_qg'

TASK_TO_FILTER_FN = {
    'qa': filter_qa,
    'qg': filter_qg,
    'e2e_qg': filter_e2e_qg,
    'ans_ext': filter_ans_ext,
    'multi': filter_multi
}

"""
Convert the SQuAD json format to dataset format (not used but is useful)
"""
def process_squad(articles):
    out = {
        "contexts": [],
        "questions": [],
        "answers": [],
    }
    for paragraphs in articles["paragraphs"]:
        for paragraph in paragraphs:
            for qa in paragraph["qas"]:
                for answer in qa["answers"]:
                    # out["title"].append(title)
                    out["contexts"].append(paragraph["context"])
                    out["questions"].append(qa["question"])
                    # out["id"].append(qa["id"])
                    out["answers"].append(answer)
    return out

"""
All three of the following function has the same semantic as the squad_multitask.py file
There will be no tags as we will always use multi - generate both questions and answers
"""
hl_token = "<hl>"            
sep_token = "<sep>"
def _get_correct_alignement(context, answer):
    """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """
    gold_text = answer['text']
    start_idx = answer['answer_start']
    end_idx = start_idx + len(gold_text)
    if context[start_idx:end_idx] == gold_text:
        return start_idx, end_idx       # When the gold label position is good
    elif context[start_idx-1:end_idx-1] == gold_text:
        return start_idx-1, end_idx-1   # When the gold label is off by one character
    elif context[start_idx-2:end_idx-2] == gold_text:
        return start_idx-2, end_idx-2   # When the gold label is off by two character
    else:
        raise ValueError()
"""
All three of the following function has the same semantic as the squad_multitask.py file
There will be no tags as we will always use multi - generate both questions and answers
"""
def process_qa_text(articles):
    out = {
        "source_text": [],
        "target_text": [],
    }
    for context, question, answer in zip(articles['contexts'], articles['questions'], articles['answers']):
        out["source_text"].append(f"question: {question}  context: {context}")
        out["target_text"].append(f"{answer}")
    return out
"""
All three of the following function has the same semantic as the squad_multitask.py file
There will be no tags as we will always use multi - generate both questions and answers
"""
def process_qg_text(articles):
    out = {
        "source_text": [],
        "target_text": [],
    }
    for context, question, answer in zip(articles['contexts'], articles['questions'], articles['answers']):
        # raise Exception((answer))
        answer_text = str(answer['text']).strip()
        start_pos, end_pos = _get_correct_alignement(context, answer)
        out["source_text"].append(f"generate question: {context[:start_pos]} {{hl_token}} {answer_text} {{hl_token}} {context[end_pos:]}")
        out["target_text"].append(f"{question}")
    return out

def main():
    parser = HfArgumentParser((DataTrainingArguments,))

    data_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    if data_args.model_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
    else:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    tokenizer.add_tokens(['<sep>', '<hl>'])

    # if load from custom script (in data/squad_multitask.py)
    if not data_args.load_local:
        train_dataset = load_dataset(data_args.dataset_path, name=data_args.qg_format, split=datasets.Split.TRAIN)
        valid_dataset = load_dataset(data_args.dataset_path, name=data_args.qg_format, split=datasets.Split.VALIDATION)
        train_dataset = train_dataset.filter(TASK_TO_FILTER_FN[data_args.task])
        if data_args.task == 'multi' and data_args.valid_for_qg_only:
            logger.info("processing valid data only for qg task")
            valid_dataset = valid_dataset.filter(filter_qg)
        else:
            valid_dataset = valid_dataset.filter(TASK_TO_FILTER_FN[data_args.task])
    else:
        # Train
        data_files_train = {"train": os.path.relpath(data_args.dataset_path + data_args.train_file)}
        # raise Exception(data_files_train)
        train_dataset = load_dataset('json', data_files=data_files_train, name=data_args.qg_format, field='data')
        train_dataset = train_dataset['train']
        try:
            train_dataset = train_dataset.remove_columns("title")
        except ValueError:
            pass
        # convert it (flatten it) to fit what we want
        train_dataset = train_dataset.map(process_squad, batched=True, remove_columns="paragraphs")
        
        # Validation
        data_files_valid = {"valid": os.path.relpath(data_args.dataset_path + data_args.valid_file)}
        valid_dataset = load_dataset('json', data_files=data_files_valid, name=data_args.qg_format, field='data')
        valid_dataset = valid_dataset['valid']
        try:
            valid_dataset = valid_dataset.remove_columns("title")
        except ValueError:
            pass
        # convert it (flatten it) to fit what we want
        valid_dataset = valid_dataset.map(process_squad, batched=True, remove_columns="paragraphs")

        # QA
        train_dataset_qa = train_dataset.map(process_qa_text, batched=True, remove_columns=['contexts', 'questions', 'answers'])
        valid_dataset_qa = train_dataset.map(process_qa_text, batched=True, remove_columns=['contexts', 'questions', 'answers'])

        # QG
        train_dataset_qg = train_dataset.map(process_qg_text, batched=True, remove_columns=['contexts', 'questions', 'answers'])
        valid_dataset_qg = train_dataset.map(process_qg_text, batched=True, remove_columns=['contexts', 'questions', 'answers'])
        
        # clear the memory
        del train_dataset
        del valid_dataset
        
        # Assign it back to the correct values
        train_dataset = concatenate_datasets([train_dataset_qa, train_dataset_qg]).shuffle()
        valid_dataset = concatenate_datasets([valid_dataset_qa, valid_dataset_qg])
      
    processor = DataProcessor(
        tokenizer,
        model_type=data_args.model_type,
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length
    )

    train_dataset = processor.process(train_dataset)
    valid_dataset = processor.process(valid_dataset)
    # train_dataset.to_json("squad2.json")
    columns = ["source_ids", "target_ids", "attention_mask"]
    train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)

    if data_args.train_file_name is None:
        train_file_name = f"train_data_{data_args.task}_{data_args.qg_format}_{data_args.model_type}.pt"
        train_path = os.path.join("data", train_file_name)

        valid_file_name = f"valid_data_{data_args.task}_{data_args.qg_format}_{data_args.model_type}.pt"
        valid_path = os.path.join("data", valid_file_name)
    else:
        train_path = os.path.join("data", data_args.train_file_name)
        valid_path = os.path.join("data", data_args.valid_file_name)
    
    torch.save(train_dataset, train_path)
    logger.info(f"saved train dataset at {train_path}")
    
    torch.save(valid_dataset, valid_path)
    logger.info(f"saved validation dataset at {valid_path}")
    
    tokenizer_path = f"{data_args.model_type}_qg_tokenizer"
    if not os.path.exists(tokenizer_path):
        os.mkdir(tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"saved tokenizer at {tokenizer_path}")


if __name__ == "__main__":
    main()