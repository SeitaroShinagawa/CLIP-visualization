import argparse
import os
import sys
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import Places365, CIFAR100
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

import clip
from clip import _transform

def enlarge_pe(pe, factor):
    """
    Making positional embedding resolution higher by using interpolation
    
    pe: (num_tokens, dim) 
    factor: factor of resolution (factor=N transforms the number of tokens into N^2 times, then patsize will be 1/N)
    """
    with torch.no_grad():
        pe_cls = pe[0]
        pe_tokens = pe[1:]
        num_tokens, dim = pe_tokens.shape
        sr_p = int(math.sqrt(num_tokens)) # srp: square root patches
        assert sr_p**2 == num_tokens

        pe_2d = pe_tokens.reshape(1,sr_p,sr_p,dim).permute(0,3,1,2) #1xCxPxP
        pe_large = torch.nn.functional.interpolate(pe_2d, scale_factor=factor, mode="bilinear") #1xCx2Px2P
        pe_new = torch.cat([pe_cls.unsqueeze(0), pe_large.permute(0,2,3,1).squeeze().reshape(-1, dim)], dim=0)
    return pe_new

def create_position_embedding_map(pe, query_id):
    """
    query_id: index of pe for comparison
    """
    query = pe[query_id]
    pe_tokens = pe[1:]
    pos_mat = pe_tokens @ query
    num_tokens, dim = pe_tokens.shape
    sr_p = int(math.sqrt(num_tokens)) # srp: square root patches
    return pos_mat.reshape(sr_p, sr_p).cpu().detach().numpy()

def subplot_figure(fig, n_h, n_w, fig_id, input_map, title=None, fontsize=22, vmin=None, vmax=None):
    ax = fig.add_subplot(n_h, n_w, fig_id)
    ax.axis("off")
    if vmin is not None and vmax is not None:
        im = ax.pcolor(input_map, norm=Normalize(vmin=vmin, vmax=vmax))
    else: 
        im = ax.pcolor(input_map)
    ax.invert_yaxis()
    if title is not None:
        plt.title(title, fontsize=fontsize) 
    if vmin is not None and vmax is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    fig.subplots_adjust(hspace=0.2, wspace=0.3)

parser = argparse.ArgumentParser(description="Visualizer of CLIP attention (average attention over heads) (option: visualize on each head, position embeddings)")
parser.add_argument("--index", default=0, type=int)
parser.add_argument("--dataset", default="place365", type=str)
parser.add_argument("--imgpath", default=None, type=str, help="use an own image. If not None, this code ignore '--index' and '--dataset'")
parser.add_argument("--savedir", default="./", type=str)
parser.add_argument("--factor", default=1, type=int, help="factor to enlarge input image size (makes position embedding resolution higher)")
parser.add_argument("--vmax", default=0.05, type=float, help="max range for output figures")
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--visualize_eachhead", action="store_true", help="visualize each head of multi-head attention")
parser.add_argument("--visualize_posemb", action="store_true", help="visualize position embeddings")
parser.add_argument("--pos_vmax", default=1.2, type=float, help="max range for output figures for position embeddings")
parser.add_argument("--pos_vmin", default=0.0, type=float, help="min range for output figures for position embeddings")
parser.add_argument("--unifyfigs", action="store_true", help="unify figures into one figure")
parser.add_argument("--layer_id", default=11, type=int, help="layer id to visualize (clip has range of 0 to 11)")
args = parser.parse_args()

p = Path(args.imgpath)
savedir = args.savedir + "/" + p.stem
print("outputs are saved to: ", savedir)
if not os.path.exists(savedir):
    os.mkdir(savedir)

device = "cuda:{}".format(args.gpu) if args.gpu >= 0 else "cpu"
model, preprocess = clip.load("ViT-B/32", jit=False, device=device)

factor=args.factor
preprocess = _transform(224*factor) # input is size of resize
model.visual.positional_embedding = torch.nn.Parameter(enlarge_pe(model.visual.positional_embedding, factor))

transform_image = Compose([
    Resize(model.visual.input_resolution, interpolation=Image.BICUBIC),
    CenterCrop(model.visual.input_resolution),
    lambda image: image.convert("RGB"),
])

# Loading an image
if args.imgpath is None:
    if args.dataset == 'place365':
        ds = Places365(root=os.path.expanduser("~/.cache"), small=True, split='val', download=True)
    elif args.dataset == 'cifar100':
        ds = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    else:
        raise ValueError
    # Prepare the inputs
    image, class_id = ds[args.index]
else:
    image = Image.open(args.imgpath).convert("RGB")

# Processing the image to get attention map
image_input = preprocess(image).unsqueeze(0).to(device)
with torch.no_grad():
    image_feat, image_attention_mh = model.encode_image_attention(image_input, return_attention=True, cond_attn=None, target_layer=args.layer_id) #N, head, query, key
    image_attention_mh = image_attention_mh.to("cpu")
# Get average attention over multi-heads
n_head = image_attention_mh.shape[1]
attention_mean = image_attention_mh[:,:,:,1:].sum(dim=1) / n_head #[:,:,:,1:] means removing cls token of key

vmax = image_attention_mh.max()
vmin = image_attention_mh.min()
print("raw attntion vmax:", vmax.item(), ", vmin:", vmin.item())

print("CLIP setting info")
print("The number of heads of multi-head attention: {}".format(n_head))
print("Resize: 224 (default) -> {}".format(224*factor))
print("Patrch size: 32 (default) -> {}".format(32//factor))
print("Figure with range 0.0-{}".format(args.vmax))
print("Figure for position embeddings with range 0.0-{}".format(args.vmax))
print("Layer id to visualize: {}".format(args.layer_id))
layer_id = args.layer_id

# Visualize all in one figure
if args.unifyfigs:
    # Figure size
    n_h = 1 + n_head//4
    n_w = 4
    fontsize=20
    
    # Creating figures on each query token
    for query_id in tqdm(range(attention_mean.shape[1]), desc="progress"):
        fig = plt.figure(figsize=[16, 16], frameon=False) 

        # Visualize input image
        ax = fig.add_subplot(n_h, n_w, 1)
        ax.axis("off")
        ax.imshow(transform_image(image))
        plt.title("input image", fontsize=fontsize)

        # Query map
        query_map = np.zeros(attention_mean.shape[-1])
        if query_id > 0: #exclude query_id = 0 because it is cls_token, not shown in query_map
            query_map[query_id-1] = 1.0
        query_map = query_map.reshape(7*factor, 7*factor)
        subplot_figure(fig, n_h, n_w, 2, 
                        query_map, 
                        title="query token", 
                        fontsize=fontsize
                        )

        # Position embeddings similarity
        pe = model.visual.positional_embedding #cls & tokens
        pos_map = create_position_embedding_map(pe, query_id)
        
        subplot_figure(fig, n_h, n_w, 3, 
                        pos_map, 
                        title="position similarity", 
                        fontsize=fontsize, 
                        vmin=args.pos_vmin, vmax=args.pos_vmax)   

        # Visualize average attention
        att_ave_map = attention_mean[0, query_id].reshape(7*factor, 7*factor)
        subplot_figure(fig, n_h, n_w, 4, 
                        att_ave_map, 
                        title="average over heads", 
                        fontsize=fontsize, 
                        vmin=0.0, vmax=args.vmax)         
        
        # Visualize atention over heads
        for head_id in range(n_head):
            att_map = image_attention_mh[0, head_id, query_id, 1:].reshape(7*factor, 7*factor)
            subplot_figure(fig, n_h, n_w, 5+head_id, 
                        att_map, 
                        title="layer{:2d}, head{:2d}".format(layer_id, head_id), 
                        fontsize=fontsize, 
                        vmin=0.0, vmax=args.vmax)   
            
        fig.savefig(savedir + "/unified_layer{:02d}_query{:02d}.png".format(layer_id, query_id))
        plt.close()

else: # Visualize each
    for query_id in range(attention_mean.shape[1]):
        # For average head
        fig = plt.figure(figsize=[10, 5], frameon=False)
        ax = fig.add_subplot(1, 2, 1)
        ax.axis("off")
        ax.imshow(transform_image(image))
        ax = fig.add_subplot(1, 2, 2)
        ax.axis("off")
        im = ax.pcolor(attention_mean[0, query_id].reshape(7*factor, 7*factor), norm=Normalize(vmin=0.0, vmax=args.vmax))
        ax.invert_yaxis()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        fig.subplots_adjust(hspace=0, wspace=0)
        fig.savefig(savedir + "/attention_layer_{:02d}_query{:02d}_average.png".format(layer_id, query_id))
        plt.close()

        # For each head
        if args.visualize_eachhead:
            fig = plt.figure(figsize=[14, 10], frameon=False)
            for head_id in range(n_head):
                ax = fig.add_subplot(n_head//4, 4, head_id+1)
                ax.axis("off")
                im = ax.pcolor(image_attention_mh[0, head_id, query_id, 1:].reshape(7*factor, 7*factor), norm=Normalize(vmin=0.0, vmax=args.vmax))
                ax.invert_yaxis()
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                fig.subplots_adjust(hspace=0.2, wspace=0.3)
            fig.savefig(savedir + "/attention_layer{:02d}_query{:02d}_eachheads.png".format(layer_id, query_id))
            plt.close()

    # Visualize position embeddings
    if args.visualize_posemb:
        pe = model.visual.positional_embedding #cls & tokens

        for query_id in range(pe.shape[0]):
            pos_mat = create_position_embedding_map(pe, query_id)
            fig = plt.figure(figsize=[5, 5], frameon=False)
            ax = fig.add_subplot(1, 1, 1)
            ax.axis("off")
            im = ax.pcolor(pos_mat, norm=Normalize(vmin=args.pos_vmin, vmax=args.pos_vmax))
            ax.invert_yaxis()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            fig.subplots_adjust(hspace=0, wspace=0)
            fig.savefig(savedir + "/posemb_query{:02d}.png".format(query_id))
            plt.close()
