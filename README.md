# USODFuseNet

### Model/Dataset Overview

| Mode | Dataset Link | Pre-computed Saliency Maps | Model Weights | Pre-trained Weights |
|------------|----------------------|----------------------------|---------------|---------------|
| USOD    | [Train](https://drive.google.com/file/d/1UFoEqDVPV7Zh7s_4kO_YkS6xxDLgB_Ve/view?usp=sharing)/[Test](https://drive.google.com/drive/folders/1v6b1Lp-naxdKbVmxsRdjkGv-QekuUetV?usp=sharing)                | [Saliency Maps](https://drive.google.com/drive/folders/10jYLlB-mKXrqockLQGZAm-UmmpOuYjQ0?usp=sharing) | [Weights](https://drive.google.com/drive/folders/1I0MdGz9x-9j0xYhFOT1f46e3p2JeNcR1?usp=sharing) | [Pre-trained Weights](https://drive.google.com/file/d/1q5JzdyZnpEWq2ur4jSJ1QNyi23O6U6J4/view?usp=sharing)
| RGBD   | [Train](https://drive.google.com/drive/folders/1MAfic5D51P55M_9Bv_RsLgN5D4re6AKU?usp=sharing)/[Test](https://drive.google.com/drive/folders/1JY90-TtVXZLHbEArXHPgKE96RDOoj2Ul?usp=sharing) | [Saliency Maps](https://drive.google.com/drive/folders/1qPRZq0mWiUipHvmKk0GlHMC5XynkqAQT?usp=sharing) | [Weights](https://drive.google.com/drive/folders/1I0MdGz9x-9j0xYhFOT1f46e3p2JeNcR1?usp=sharing) | [Pre-trained Weights](https://drive.google.com/file/d/1q5JzdyZnpEWq2ur4jSJ1QNyi23O6U6J4/view?usp=sharing)

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
    --use_pretrained 1 \
    --checkpoint_name "SODAWideNet++" \ 
    --im_size 256
```

For RGBD training, use the following command - 

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
    --use_pretrained 1 \
    --checkpoint_name "SODAWideNet++" \ 
    --im_size 256
```

### Inference

Download the trained weights and place them in a folder named _checkpoints_. Also, change checkpoint names in _inference.py_ file (line numbers 24 & 25) for different modes.

1. Use the command below to produce a saliency map for a single image-depth pair. Remove the _display_ flag from the above command to save the prediction.

```bash
python inference.py --mode single --input_path path/to/image.jpg --depth_path path/to/depth_image.jpg --display
``` 

2. Use the below command to run inference on a series of images in a folder and save the predicted saliency maps.

```bash
python inference.py --mode folder --input_path path/to/image_folder --depth_path path/to/depth_folder --output_dir path/to/output_folder
```
