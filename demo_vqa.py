import gradio as gr
import torch
import copy
import time
import requests
import io
import numpy as np
import re
import json
import urllib.request

import ipdb

from PIL import Image

from vilt.config import ex
from vilt.modules import ViLTransformerSS

from vilt.transforms import pixelbert_transform
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)

    loss_names = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 1,
        "imgcls": 0,
        "nlvr2": 0,
        "irtr": 0,
        "arc": 0,
    }
    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])

    with urllib.request.urlopen(
        "https://github.com/dandelin/ViLT/releases/download/200k/vqa_dict.json"
    ) as url:
        id2ans = json.loads(url.read().decode())

    _config.update(
        {
            "loss_names": loss_names,
        }
    )

    model = ViLTransformerSS(_config)
    model.setup("test")
    model.eval()

    device = "cuda:0" if _config["num_gpus"] > 0 else "cpu"
    model.to(device)

    def infer(url, text):
        try:
            res = requests.get(url)
            image = Image.open(io.BytesIO(res.content)).convert("RGB")
            img = pixelbert_transform(size=384)(image)
            img = img.unsqueeze(0).to(device)
        except:
            return False

        batch = {"text": [text], "image": [img]}

        with torch.no_grad():
            encoded = tokenizer(batch["text"])
            batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
            batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
            batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
            infer = model.infer(batch)
            vqa_logits = model.vqa_classifier(infer["cls_feats"])

        answer = id2ans[str(vqa_logits.argmax().item())]

        return [np.array(image), answer]

    inputs = [
        gr.inputs.Textbox(
            label="Url of an image.",
            lines=5,
        ),
        gr.inputs.Textbox(label="Question", lines=5),
    ]
    outputs = [
        gr.outputs.Image(label="Image"),
        gr.outputs.Textbox(label="Answer"),
    ]

    interface = gr.Interface(
        fn=infer,
        inputs=inputs,
        outputs=outputs,
        server_name="0.0.0.0",
        server_port=8888,
        examples=[
            [
                "https://s3.geograph.org.uk/geophotos/06/21/24/6212487_1cca7f3f_1024x1024.jpg",
                "What is the color of the flower?",
            ],
            [
                "https://computing.ece.vt.edu/~harsh/visualAttention/ProjectWebpage/Figures/vqa_1.png",
                "What is the mustache made of?",
            ],
            [
                "https://computing.ece.vt.edu/~harsh/visualAttention/ProjectWebpage/Figures/vqa_2.png",
                "How many slices of pizza are there?",
            ],
            [
                "https://computing.ece.vt.edu/~harsh/visualAttention/ProjectWebpage/Figures/vqa_3.png",
                "Does it appear to be rainy?",
            ],
        ],
    )

    interface.launch(debug=True)
