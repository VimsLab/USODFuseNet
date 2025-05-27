# USODFuseNet

### Training

There are two settings: USOD (Underwater Salient Object Detection) and RGBD. Below, changing the _training_scheme_ will enable either training.

The below command for USOD training - 

```bash
python training.py \
    --lr 0.0005 \
    --epochs 26 \
    --f_name "USODFuseNet" \
    --n 4 \
    --b 16 \
    --sched 1 \
    --training_scheme "RGBD" \
    --salient_loss_weight 1.0 \
    --use_pretrained 0 \
    --im_size 256
```


```bash
python training.py \
    --lr 0.0005 \
    --epochs 21 \
    --f_name "USODFuseNetRGBD" \
    --n 4 \
    --b 16 \
    --sched 1 \
    --training_scheme "RGBD" \
    --salient_loss_weight 1.0 \
    --use_pretrained 0 \
    --im_size 256
```
