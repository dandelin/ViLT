# Evaluation
The results will vary a bit since we do a batched-inference, which yields padded image batch that would be inconsistently embedded while performing linear image patch projection.

## Evaluate VQAv2
```bash
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> per_gpu_batchsize=<BS_FITS_YOUR_GPU> task_finetune_vqa_randaug test_only=True precision=32 load_path="<YOUR_WEIGHT_ROOT>/vilt_vqa.ckpt"

ex)
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 per_gpu_batchsize=64 task_finetune_vqa_randaug test_only=True precision=32 load_path="weights/vilt_vqa.ckpt"

output > This script will generate `result/vqa_submit_vilt_vqa.json`, you can upload it to eval.ai (https://eval.ai/web/challenges/challenge-page/830/overview) evaluation server to get test-dev score.
[{"test-dev": {"yes/no": 87.44, "number": 50.2, "other": 62.38, "overall": 71.32}}]
```

## Evaluate NLVR2
```bash
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> per_gpu_batchsize=<BS_FITS_YOUR_GPU> task_finetune_nlvr2_randaug test_only=True precision=32 load_path="<YOUR_WEIGHT_ROOT>/vilt_nlvr2.ckpt"

ex)
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 per_gpu_batchsize=64 task_finetune_nlvr2_randaug test_only=True precision=32 load_path="weights/vilt_nlvr2.ckpt"

output >
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'nlvr2/dev/accuracy': tensor(0.7486, device='cuda:0'),
 'nlvr2/dev/accuracy_epoch': tensor(0.7565, device='cuda:0'),
 'nlvr2/dev/loss': tensor(0.8581, device='cuda:0'),
 'nlvr2/dev/loss_epoch': tensor(0.8609, device='cuda:0'),
 'nlvr2/test/accuracy': tensor(0.7735, device='cuda:0'),
 'nlvr2/test/accuracy_epoch': tensor(0.7652, device='cuda:0'),
 'nlvr2/test/loss': tensor(0.7796, device='cuda:0'),
 'nlvr2/test/loss_epoch': tensor(0.8381, device='cuda:0'),
 'val/the_metric': tensor(0.7652, device='cuda:0')}
--------------------------------------------------------------------------------
INFO - ViLT - Completed after 0:01:31
```

## Evaluate COCO IR/TR
```bash
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> per_gpu_batchsize=<BS_FITS_YOUR_GPU> task_finetune_irtr_coco_randaug test_only=True precision=32 load_path="<YOUR_WEIGHT_ROOT>/vilt_irtr_coco.ckpt"

or you can evaluate zero-shot performance just simply using "<YOUR_WEIGHT_ROOT>/vilt_200k_mlm_itm.ckpt" instead.

ex)
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 per_gpu_batchsize=4 task_finetune_irtr_coco_randaug test_only=True precision=32 load_path="weights/vilt_irtr_coco.ckpt"

output > caution! this will take a lot of time (= run transformer for 5000 x 5000 samples; the returned values are IR R@1, R@5, R@10 and TR R@1, R@5, R@10)
(tensor(0.4299), tensor(0.7284), tensor(0.8307), tensor(0.6162), tensor(0.8632), tensor(0.9270)) 0
Testing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [34:58:50<00:00, 161.04s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'irtr/val/irtr_loss': tensor(0.0533, device='cuda:0'),
 'irtr/val/irtr_loss_epoch': tensor(0.0540, device='cuda:0'),
 'val/the_metric': 1.0460796058177948}
--------------------------------------------------------------------------------
INFO - ViLT - Completed after 1 day, 10:59:12
```

## Evaluate F30K IR/TR
```bash
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> per_gpu_batchsize=<BS_FITS_YOUR_GPU> task_finetune_irtr_f30k_randaug test_only=True precision=32 load_path="<YOUR_WEIGHT_ROOT>/vilt_irtr_f30k.ckpt"

or you can evaluate zero-shot performance just simply using "<YOUR_WEIGHT_ROOT>/vilt_200k_mlm_itm.ckpt" instead.

ex)
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 per_gpu_batchsize=4 task_finetune_irtr_f30k_randaug test_only=True precision=32 load_path="weights/vilt_irtr_f30k.ckpt"

output > the returned values are IR R@1, R@5, R@10 and TR R@1, R@5, R@10)
(tensor(0.6454), tensor(0.8886), tensor(0.9392), tensor(0.8360), tensor(0.9680), tensor(0.9860)) 0
Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [1:26:35<00:00, 33.09s/it]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'irtr/val/irtr_loss': tensor(0.1331, device='cuda:0'),
 'irtr/val/irtr_loss_epoch': tensor(0.1189, device='cuda:0'),
 'val/the_metric': 1.4814000129699707}
--------------------------------------------------------------------------------
INFO - ViLT - Completed after 1:27:01
```
