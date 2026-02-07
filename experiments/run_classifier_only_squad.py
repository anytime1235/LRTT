#!/usr/bin/env python
"""
Classifier-only finetuning on SQuAD.
Freezes all layers except qa_outputs (the QA classification head).
Uses the same data pipeline as run_qa.py.

Usage:
    cd /data/LRTT_transformer/lora_training
    python ../experiments/run_classifier_only_squad.py \
        --model_name_or_path google/mobilebert-uncased \
        --dataset_name squad \
        --do_train --do_eval \
        --per_device_train_batch_size 32 \
        --num_train_epochs 3 \
        --logging_steps 50 \
        --save_strategy no \
        --output_dir ./results/squad_classifier_only
"""
import sys
from pathlib import Path

# Add lora_training to path so we can import trainer_qa, utils_qa
lora_training_dir = str(Path(__file__).resolve().parent.parent / "lora_training")
if lora_training_dir not in sys.path:
    sys.path.insert(0, lora_training_dir)

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
from datasets import load_dataset
from trainer_qa import QuestionAnsweringTrainer
from utils_qa import postprocess_qa_predictions

import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    model_name_or_path: str = field(default="google/mobilebert-uncased")
    dataset_name: str = field(default="squad")
    max_seq_length: int = field(default=384)
    doc_stride: int = field(default=128)
    pad_to_max_length: bool = field(default=True)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    n_best_size: int = field(default=20)
    max_answer_length: int = field(default=30)


def main():
    parser = HfArgumentParser((TrainingArguments, DataArguments))
    training_args, extra_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    set_seed(training_args.seed)

    # Load dataset
    raw_datasets = load_dataset(extra_args.dataset_name)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(extra_args.model_name_or_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(extra_args.model_name_or_path)

    # =========================================
    # FREEZE ALL LAYERS EXCEPT qa_outputs
    # =========================================
    trainable_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        total_count += param.numel()
        if "qa_outputs" in name:
            param.requires_grad = True
            trainable_count += param.numel()
            print(f"  [TRAINABLE] {name}: {param.shape}")
        else:
            param.requires_grad = False

    print(f"\n{'='*50}")
    print(f"CLASSIFIER-ONLY FINETUNING (qa_outputs)")
    print(f"{'='*50}")
    print(f"Trainable: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.4f}%)")
    print(f"{'='*50}\n")

    # Preprocessing (same as run_qa.py)
    column_names = raw_datasets["train"].column_names
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"

    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(extra_args.max_seq_length, tokenizer.model_max_length)

    def prepare_train_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        tokenized = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=extra_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if extra_args.pad_to_max_length else False,
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        tokenized["start_positions"] = []
        tokenized["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized.sequence_ids(i)

            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            if len(answers["answer_start"]) == 0:
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized["start_positions"].append(cls_index)
                    tokenized["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized["end_positions"].append(token_end_index + 1)

        return tokenized

    train_dataset = raw_datasets["train"].map(
        prepare_train_features, batched=True, remove_columns=column_names,
        desc="Running tokenizer on train dataset",
    )
    if extra_args.max_train_samples:
        train_dataset = train_dataset.select(range(extra_args.max_train_samples))

    def prepare_validation_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        tokenized = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=extra_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if extra_args.pad_to_max_length else False,
        )
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["example_id"] = []
        for i in range(len(tokenized["input_ids"])):
            sequence_ids = tokenized.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            sample_index = sample_mapping[i]
            tokenized["example_id"].append(examples["id"][sample_index])
            tokenized["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized["offset_mapping"][i])
            ]
        return tokenized

    eval_examples = raw_datasets["validation"]
    if extra_args.max_eval_samples:
        eval_examples = eval_examples.select(range(extra_args.max_eval_samples))
    eval_dataset = eval_examples.map(
        prepare_validation_features, batched=True, remove_columns=column_names,
        desc="Running tokenizer on validation dataset",
    )

    data_collator = default_data_collator if extra_args.pad_to_max_length else DataCollatorWithPadding(tokenizer)

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    def post_processing_function(examples, features, predictions, stage="eval"):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=False,
            n_best_size=extra_args.n_best_size,
            max_answer_length=extra_args.max_answer_length,
            null_score_diff_threshold=0.0,
            output_dir=training_args.output_dir,
            log_level=log_level,
            prefix=stage,
        )
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Train
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    # Evaluate
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        print(f"\n{'='*50}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*50}")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
