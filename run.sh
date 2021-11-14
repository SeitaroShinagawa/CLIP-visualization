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
