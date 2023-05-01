from vilt.datasets import GQADataset
from .datamodule_base import BaseDataModule
from collections import defaultdict
import numpy as np

class GQADataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return GQADataset

    @property
    def dataset_name(self):
        return "gqa"

    def setup(self, stage):
        super().setup(stage)

        train_answers = self.train_dataset.table["answers"].to_pandas().tolist()
        val_answers = self.val_dataset.table["answers"].to_pandas().tolist()
        train_labels = self.train_dataset.table["answer_label"].to_pandas().tolist()
        val_labels = self.val_dataset.table["answer_label"].to_pandas().tolist()

        all_answers = [c for c in train_answers + val_answers if c is not None]

        train_answer_tuples = [(label, answer) for labels, answers in zip(train_labels, train_answers) for label, answer in zip(labels.tolist(), answers.tolist())]
        val_answer_tuples = [(label, answer) for labels, answers in zip(val_labels, val_answers) for label, answer in zip(labels.tolist(), answers.tolist())]
        
        train_answer2id = {answer: label for label, answer in train_answer_tuples}
        val_answer2id = {answer: label for label, answer in val_answer_tuples}
        # print([i for i in train_answer2id if train_answer2id[i]==2])
        # Merge train and val dictionaries, keeping the label ids from the train dictionary
        self.answer2id = {**val_answer2id, **train_answer2id}
        
        self.num_class = len(self.answer2id)
        self.id2answer = defaultdict(lambda: "unknown")
        for k, v in self.answer2id.items():
            self.id2answer[v] = k

        # Print some samples from the training dataset
        
        # print("Training dataset samples:")
        # for idx, sample in enumerate(self.train_dataset):
        #     if idx >= 10:
        #         break
        #     print('In GQADataModule')
        #     question = sample["text"]
        #     label = sample["gqa_label"]
        #     answer = self.id2answer[label]
        #     print(f"Question: {question}\nLabel: {label}\nAnswer: {answer}")

        # print("\nValidation dataset samples:")
        # # Print some samples from the validation dataset
        # for idx, sample in enumerate(self.val_dataset):
        #     if idx >= 10:
        #         break
        #     print('In GQADataModule')
        #     question = sample["text"]
        #     label = sample["gqa_label"]
        #     answer = self.id2answer[label]
        #     print(f"Question: {question}\nLabel: {label}\nAnswer: {answer}")