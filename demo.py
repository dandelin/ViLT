import gradio as gr
import torch
import copy
import time
import requests
import io
import numpy as np
import re

import ipdb

from PIL import Image

from vilt.config import ex
from vilt.modules import ViLTransformerSS

from vilt.modules.objectives import cost_matrix_cosine, ipot
from vilt.transforms import pixelbert_transform
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)

    loss_names = {
        "itm": 0,
        "mlm": 0.5,
        "mpp": 0,
        "vqa": 0,
        "imgcls": 0,
        "nlvr2": 0,
        "irtr": 0,
        "arc": 0,
    }
    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])

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

    def infer(url, mp_text, hidx):
        try:
            res = requests.get(url)
            image = Image.open(io.BytesIO(res.content)).convert("RGB")
            img = pixelbert_transform(size=384)(image)
            img = img.unsqueeze(0).to(device)
        except:
            return False

        batch = {"text": [""], "image": [None]}
        tl = len(re.findall("\[MASK\]", mp_text))
        inferred_token = [mp_text]
        batch["image"][0] = img

        with torch.no_grad():
            for i in range(tl):
                batch["text"] = inferred_token
                encoded = tokenizer(inferred_token)
                batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
                batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
                batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
                encoded = encoded["input_ids"][0][1:-1]
                infer = model(batch)
                mlm_logits = model.mlm_score(infer["text_feats"])[0, 1:-1]
                mlm_values, mlm_ids = mlm_logits.softmax(dim=-1).max(dim=-1)
                mlm_values[torch.tensor(encoded) != 103] = 0
                select = mlm_values.argmax().item()
                encoded[select] = mlm_ids[select].item()
                inferred_token = [tokenizer.decode(encoded)]

        selected_token = ""
        encoded = tokenizer(inferred_token)

        if hidx > 0 and hidx < len(encoded["input_ids"][0][:-1]):
            with torch.no_grad():
                batch["text"] = inferred_token
                batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
                batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
                batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
                infer = model(batch)
                txt_emb, img_emb = infer["text_feats"], infer["image_feats"]
                txt_mask, img_mask = (
                    infer["text_masks"].bool(),
                    infer["image_masks"].bool(),
                )
                for i, _len in enumerate(txt_mask.sum(dim=1)):
                    txt_mask[i, _len - 1] = False
                txt_mask[:, 0] = False
                img_mask[:, 0] = False
                txt_pad, img_pad = ~txt_mask, ~img_mask

                cost = cost_matrix_cosine(txt_emb.float(), img_emb.float())
                joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
                cost.masked_fill_(joint_pad, 0)

                txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(
                    dtype=cost.dtype
                )
                img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(
                    dtype=cost.dtype
                )
                T = ipot(
                    cost.detach(),
                    txt_len,
                    txt_pad,
                    img_len,
                    img_pad,
                    joint_pad,
                    0.1,
                    1000,
                    1,
                )

                plan = T[0]
                plan_single = plan * len(txt_emb)
                cost_ = plan_single.t()

                cost_ = cost_[hidx][1:].cpu()

                patch_index, (H, W) = infer["patch_index"]
                heatmap = torch.zeros(H, W)
                for i, pidx in enumerate(patch_index[0]):
                    h, w = pidx[0].item(), pidx[1].item()
                    heatmap[h, w] = cost_[i]

                heatmap = (heatmap - heatmap.mean()) / heatmap.std()
                heatmap = np.clip(heatmap, 1.0, 3.0)
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

                _w, _h = image.size
                overlay = Image.fromarray(np.uint8(heatmap * 255), "L").resize(
                    (_w, _h), resample=Image.NEAREST
                )
                image_rgba = image.copy()
                image_rgba.putalpha(overlay)
                image = image_rgba

                selected_token = tokenizer.convert_ids_to_tokens(
                    encoded["input_ids"][0][hidx]
                )

        return [np.array(image), inferred_token[0], selected_token]

    inputs = [
        gr.inputs.Textbox(
            label="Url of an image.",
            lines=5,
        ),
        gr.inputs.Textbox(label="Caption with [MASK] tokens to be filled.", lines=5),
        gr.inputs.Slider(
            minimum=0,
            maximum=38,
            step=1,
            label="Index of token for heatmap visualization (ignored if zero)",
        ),
    ]
    outputs = [
        gr.outputs.Image(label="Image"),
        gr.outputs.Textbox(label="description"),
        gr.outputs.Textbox(label="selected token"),
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
                "a display of flowers growing out and over the [MASK] [MASK] in front of [MASK] on a [MASK] [MASK].",
                0,
            ],
            [
                "https://s3.geograph.org.uk/geophotos/06/21/24/6212487_1cca7f3f_1024x1024.jpg",
                "a display of flowers growing out and over the retaining wall in front of cottages on a cloudy day.",
                4,
            ],
            [
                "https://s3.geograph.org.uk/geophotos/06/21/24/6212487_1cca7f3f_1024x1024.jpg",
                "a display of flowers growing out and over the retaining wall in front of cottages on a cloudy day.",
                11,
            ],
            [
                "https://s3.geograph.org.uk/geophotos/06/21/24/6212487_1cca7f3f_1024x1024.jpg",
                "a display of flowers growing out and over the retaining wall in front of cottages on a cloudy day.",
                15,
            ],
            [
                "https://s3.geograph.org.uk/geophotos/06/21/24/6212487_1cca7f3f_1024x1024.jpg",
                "a display of flowers growing out and over the retaining wall in front of cottages on a cloudy day.",
                18,
            ],
            [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Living_Room.jpg/800px-Living_Room.jpg",
                "a room with a [MASK], a [MASK], a [MASK], and a [MASK].",
                0,
            ],
            [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Living_Room.jpg/800px-Living_Room.jpg",
                "a room with a rug, a chair, a painting, and a plant.",
                5,
            ],
            [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Living_Room.jpg/800px-Living_Room.jpg",
                "a room with a rug, a chair, a painting, and a plant.",
                8,
            ],
            [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Living_Room.jpg/800px-Living_Room.jpg",
                "a room with a rug, a chair, a painting, and a plant.",
                11,
            ],
            [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Living_Room.jpg/800px-Living_Room.jpg",
                "a room with a rug, a chair, a painting, and a plant.",
                15,
            ],
        ],
    )

    interface.launch(debug=True)