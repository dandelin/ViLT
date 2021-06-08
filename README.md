# ViLT

Code for the ICML 2021 (long talk) paper: "[ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334)"

---
<p align="center">
  <img align="middle" src="./assets/vilt.png" alt="The main figure"/>
</p>

## Install
```bash
pip install -r requirements.txt
pip install -e .
```

## Download Pretrained Weights
We provide five pretrained weights
1. ViLT-B/32 Pretrained with MLM+ITM for 200k steps on GCC+SBU+COCO+VG (ViLT-B/32 200k) [link](https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt)
2. ViLT-B/32 200k finetuned on VQAv2 [link](https://github.com/dandelin/ViLT/releases/download/200k/vilt_vqa.ckpt)
3. ViLT-B/32 200k finetuned on NLVR2 [link](https://github.com/dandelin/ViLT/releases/download/200k/vilt_nlvr2.ckpt)
4. ViLT-B/32 200k finetuned on COCO IR/TR [link](https://github.com/dandelin/ViLT/releases/download/200k/vilt_irtr_coco.ckpt)
5. ViLT-B/32 200k finetuned on F30K IR/TR [link](https://github.com/dandelin/ViLT/releases/download/200k/vilt_irtr_f30k.ckpt)

## Out-of-the-box MLM + Visualization Demo
<p align="center">
  <img align="middle" src="./assets/mlm.png" alt="MLM + Visualization"/>
</p>

```bash
pip install gradio==1.6.4
python demo.py with num_gpus=<0 if you have no gpus else 1> load_path="<YOUR_WEIGHT_ROOT>/vilt_200k_mlm_itm.ckpt"

ex)
python demo.py with num_gpus=0 load_path="weights/vilt_200k_mlm_itm.ckpt"
```

## Out-of-the-box VQA Demo
<p align="center">
  <img align="middle" src="./assets/vqa.png" alt="VQA"/>
</p>

```bash
pip install gradio==1.6.4
python demo_vqa.py with num_gpus=<0 if you have no gpus else 1> load_path="<YOUR_WEIGHT_ROOT>/vilt_vqa.ckpt" test_only=True

ex)
python demo_vqa.py with num_gpus=0 load_path="weights/vilt_vqa.ckpt" test_only=True
```

## Dataset Preparation
See [`DATA.md`](DATA.md)

## Train New Models
See [`TRAIN.md`](TRAIN.md)

## Evaluation
See [`EVAL.md`](EVAL.md)

## Citation
If you use any part of this code and pretrained weights for your own purpose, please cite our [paper](https://arxiv.org/abs/2102.03334).
```
@article{kim2021vilt,
  title={ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision},
  author={Kim, Wonjae and Son, Bokyung and Kim, Ildoo},
  journal={arXiv preprint arXiv:2102.03334},
  year={2021}
}
```

## Contact for Issues
- [Wonjae Kim](https://wonjae.kim/)
- [Bokyung Son](https://bo-son.github.io/)
- [Ildoo Kim](https://www.linkedin.com/in/ildoo-kim-56962034/)