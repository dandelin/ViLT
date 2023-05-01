import json
import pandas as pd
import pyarrow as pa
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict

def get_score():
    return 1.0

def path2rest(path, split, annotations, image_id_str, answer_label_dict):
    with open(path, "rb") as fp:
        binary = fp.read()

    _annot = annotations[split][image_id_str]
    _annot = list(_annot.items())
    qids, qas = [a[0] for a in _annot], [a[1] for a in _annot]
    questions = [qa[0] for qa in qas]
    if split == "train" or split == "val":
        answers = [qa[1] for qa in qas]
        answer_label = (
        [answer_label_dict.setdefault(answer, len(answer_label_dict) + 1) for answer in answers]
        )
    else:
        answers = []
        answer_label=[]

    answer_scores = [get_score() for _ in answers] if "test" not in split else list()

    return [binary, questions, answers, answer_label, answer_scores, image_id_str, qids, split]

def split_val_dataset(dataset_root, arrow_filename, split_ratio=0.9):
    table = pa.ipc.RecordBatchFileReader(
        pa.memory_map(f"{dataset_root}/{arrow_filename}", "r")
    ).read_all()

    pdtable = table.to_pandas()

    split_index = int(len(pdtable) * split_ratio)
    df1 = pdtable[:split_index]
    df2 = pdtable[split_index:]

    df1 = pa.Table.from_pandas(df1)
    df2 = pa.Table.from_pandas(df2)

    with pa.OSFile(f"{dataset_root}/gqa_trainable_val.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, df1.schema) as writer:
            writer.write_table(df1)

    with pa.OSFile(f"{dataset_root}/gqa_rest_val.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, df2.schema) as writer:
            writer.write_table(df2)


def make_arrow(root, dataset_root):
    # Read question files
    answer_label_dict = dict()
    answer_label_counter = 1
    question_files = {
        "train": [f"{root}/questions1.2/train_balanced_questions.json"],
        "val": [f"{root}/questions1.2/val_balanced_questions.json"],
        "test": [f"{root}/questions1.2/test_balanced_questions.json"],
        "testdev": [f"{root}/questions1.2/testdev_all_questions.json"]
    }

    annotations = dict()

    # for split in ["train", "val"]:
    for split in ["test", "testdev"]:
        _annot = defaultdict(dict)
        for question_file in question_files[split]:
            with open(question_file, "r") as fp:
                questions = json.load(fp)
            for q_id, q in tqdm(questions.items()):
                if split == "test":
                    _annot[q["imageId"]][q_id] = [q["question"]]
                else: 
                    _annot[q["imageId"]][q_id] = [q["question"], q["answer"]]

        annotations[split] = _annot

    # for split in ["train", "val"]:
    for split in ["test", "testdev"]:
        paths = list(glob(f"{root}/images/*.jpg"))
        annot_paths=[]
        for path in paths:
            image_id_str = path.split("/")[-1].split(".")[0]
            if image_id_str in annotations[split]:
                annot_paths.append(path)

        print(f'{split}: {len(paths)}, {len(annot_paths)}, {len(annotations[split])}')

        bs = []
        for path in tqdm(annot_paths):
            image_id_str = path.split("/")[-1].split(".")[0]
            bs.append(path2rest(path, split, annotations, image_id_str, answer_label_dict))

        dataframe = pd.DataFrame(
            bs,
            columns=[
                "image",
                "questions",
                "answers",
                "answer_label",  # Add answer_label here
                "answer_scores",
                "image_id",
                "question_id",
                "split",
            ],
        )
        print(dataframe['questions'].iloc[:5])
        print(dataframe['answers'].iloc[:5])
        print(dataframe['answer_label'].iloc[:5])
        print(dataframe['answer_scores'].iloc[:5])
        print(dataframe['image_id'].iloc[:5])
        print(dataframe['question_id'].iloc[:5])
        print(answer_label_dict)

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/gqa_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        
        split_val_dataset(dataset_root, "gqa_val.arrow")