#!/usr/bin/env python

import time
from pathlib import Path

import streamlit as st
from PIL import Image


st.set_page_config(page_title="CLIP Visualizer",layout='wide')

st.title("CLIP Visualizer")
st.header("Support ViT-B/32 Multi-head attention visualization")
st.subheader("query=0 represents CLS token as the target of attention")


src_root_dir = "./output_examples"
p = Path(src_root_dir)
src_list = list(p.glob("*"))

src_dir = st.selectbox("source directory", src_list)
src_dir = str(src_dir)

N_layers = 12
N_patch = 50

# index is start from zero
N_layers -= 1
N_patch -= 1

# image width seems not controllable
#img_width = st.slider('Image size', min_value=320, max_value=1280, step=320)
img_width=640

timestep_sleep = 0.1

#def sweep(name, num_items, fixed_value, value, ph, timestep_sleep, ph_image):
#    if st.button("{} sweep".format(name)):
#        for x in range(num_items-value):
#            time.sleep(timestep_sleep)
#            value = ph.slider("{}: (0-{})".format(name, num_items), min_value=0, max_value=num_items-1, value=value+1, step=1, key="{}_anime".format(name))
#        
#            image = Image.open(src_dir+"/"+"unified_layer{:02d}_query{:02d}.png".format(layer2, query2))
#            ph_image.image(
#                image,
#                width=img_width,
#                use_column_width="auto",
#                )

col1, col2 = st.columns(2)
with col1:
    layer1_ph = st.empty()
    layer1 = layer1_ph.slider('layer (0-11)', min_value=0, max_value=N_layers-1, value=0, step=1, key="layer1")
    query1_ph = st.empty()
    query1 = query1_ph.slider('query (0-N_patch^2)', min_value=0, max_value=N_patch-1, value=0, step=1, key="query1")
    
    placeholder1 = st.empty()
    image1 = Image.open(src_dir+"/"+"unified_layer{:02d}_query{:02d}.png".format(layer1, query1))
    placeholder1.image(
        image1, caption="Visualization with layer:{:02d}, query:{:02d}".format(layer1, query1),
        width=img_width,
        use_column_width="auto",
        )
    
with col2:
    layer2_ph = st.empty()
    layer2 = layer2_ph.slider('layer (0-11)', min_value=0, max_value=N_layers-1, value=0, step=1, key="layer2")
    query2_ph = st.empty()
    query2 = query2_ph.slider('query (0-N_patch^2)', min_value=0, max_value=N_patch-1, value=0, step=1, key="query2")
    
    placeholder2 = st.empty()
    image2 = Image.open(src_dir+"/"+"unified_layer{:02d}_query{:02d}.png".format(layer2, query2))
    placeholder2.image(
        image2, caption="Visualization with layer:{:02d}, query:{:02d}".format(layer2, query2),
        width=img_width,
        use_column_width="auto",
        )


    #sweep("layer2", N_layers, query2, layer2, layer2_ph, 0.1, placeholder)
    #sweep("query2", N_patch, layer2, query2, query2_ph, 0.1, placeholder)

    if st.button('layer sweep (right)'):
        for x in range(N_layers-layer2):
            time.sleep(timestep_sleep)
            layer2 = layer2_ph.slider('layer (0-11)', min_value=0, max_value=N_layers-1, value=layer2+1, step=1, key="layer2_anime")
            image2 = Image.open(src_dir+"/"+"unified_layer{:02d}_query{:02d}.png".format(layer2, query2))
            placeholder2.image(
                image2, caption="Visualization with layer:{:02d}, query:{:02d}".format(layer2, query2),
                width=img_width,
                use_column_width="auto",
                )
    if st.button('query sweep (right)'):
        for x in range(N_patch-query2):
            time.sleep(timestep_sleep)
            query2 = query2_ph.slider('query (0-N_patch^2)', min_value=0, max_value=N_patch-1, value=query2+1, step=1, key="query2_anime")
            image2 = Image.open(src_dir+"/"+"unified_layer{:02d}_query{:02d}.png".format(layer2, query2))
            placeholder2.image(
                image2, caption="Visualization with layer:{:02d}, query:{:02d}".format(layer2, query2),
                width=img_width,
                use_column_width="auto",
                )
    
    if st.button('layer sweep (tied)'):
        layer1 = layer1_ph.slider('layer (0-11)', min_value=0, max_value=N_layers-1, value=0, step=1, key="layer1_tied")
        layer2 = layer2_ph.slider('layer (0-11)', min_value=0, max_value=N_layers-1, value=0, step=1, key="layer2_tied")
        for x in range(N_layers):
            time.sleep(timestep_sleep)
            layer1 = layer1_ph.slider('layer (0-11)', min_value=0, max_value=N_layers-1, value=layer1+1, step=1, key="layer1_tied_anime")
            
            image1 = Image.open(src_dir+"/"+"unified_layer{:02d}_query{:02d}.png".format(layer1, query1))
            placeholder1.image(
                image1, caption="Visualization with layer:{:02d}, query:{:02d}".format(layer1, query1),
                width=img_width,
                use_column_width="auto",
                )
            layer2 = layer2_ph.slider('layer (0-11)', min_value=0, max_value=N_layers-1, value=layer2+1, step=1, key="layer2_tied_anime")
            
            image2 = Image.open(src_dir+"/"+"unified_layer{:02d}_query{:02d}.png".format(layer2, query2))
            placeholder2.image(
                image2, caption="Visualization with layer:{:02d}, query:{:02d}".format(layer2, query2),
                width=img_width,
                use_column_width="auto",
                )

    if st.button('query sweep (tied)'):
        query1 = query1_ph.slider('query (0-N_patch^2)', min_value=0, max_value=N_patch-1, value=0, step=1, key="query1_tied")
        query2 = query2_ph.slider('query (0-N_patch^2)', min_value=0, max_value=N_patch-1, value=0, step=1, key="query2_tied")
        for x in range(N_patch-query1):
            time.sleep(timestep_sleep)
            query1 = query1_ph.slider('query (0-N_patch^2)', min_value=0, max_value=N_patch-1, value=query1+1, step=1, key="query1_tied_anime")
        
            image1 = Image.open(src_dir+"/"+"unified_layer{:02d}_query{:02d}.png".format(layer1, query1))
            placeholder1.image(
                image1, caption="Visualization with layer:{:02d}, query:{:02d}".format(layer1, query1),
                width=img_width,
                use_column_width="auto",
                )
            
            query2 = query2_ph.slider('query (0-N_patch^2)', min_value=0, max_value=N_patch-1, value=query2+1, step=1, key="query2_tied_anime")
        
            image2 = Image.open(src_dir+"/"+"unified_layer{:02d}_query{:02d}.png".format(layer2, query2))
            placeholder2.image(
                image2, caption="Visualization with layer:{:02d}, query:{:02d}".format(layer2, query2),
                width=img_width,
                use_column_width="auto",
                )
            
