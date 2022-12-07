# CLIP-visualization
Attention visualization in CLIP
- Google Colab: https://colab.research.google.com/drive/1lBER0KCDQOuLKwVQRptQJNoWv8Y-Fun3?usp=sharing

base: https://github.com/openai/CLIP

- Main script is `visualize_clip_attentions.py`
- Use `python visualize_clip_attentions.py --help` to see the options.
- e.g. run `bash run.sh`

Visualizer of CLIP attention (average attention over heads) (option: visualize on each head, position embeddings)  
```
optional arguments:  
  -h, --help            show this help message and exit
  --index INDEX
  --dataset DATASET
  --imgpath IMGPATH     use an own image. If not None, this code ignore '--index' and '--dataset'
  --savedir SAVEDIR
  --factor FACTOR       factor to enlarge input image size (makes position embedding resolution higher)
  --vmax VMAX           max range for output figures
  --gpu GPU
  --visualize_eachhead  visualize each head of multi-head attention
  --visualize_posemb    visualize position embeddings
  --pos_vmax POS_VMAX   max range for output figures for position embeddings
  --pos_vmin POS_VMIN   min range for output figures for position embeddings
  --unifyfigs           unify figures into one figure
  --layer_id LAYER_ID   layer id to visualize (clip has range of 0 to 11)
```


run.sh
```
#!/bin/bash

# layer-wise visualization
for i in `seq 0 11`;
  do python visualize_clip_attentions.py \
    --imgpath "src_examples/sample_human.png" \
    --savedir "output_examples" \
    --vmax 0.05 \
    --unifyfigs \
    --layer_id $i \
  ;done

```
