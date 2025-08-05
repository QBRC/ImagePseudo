import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os
import argparse
import pandas as pd
import skimage
import skimage.io
import torch
from torchvision import transforms
from transformers import ViTModel
from openslide import open_slide, ImageSlide
import numexpr as ne
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from segmentation_functions import get_mask_for_slide_image

#### Define tumor grade classes ####
CLASSES = {"G1",
           "G2",
           "G3",
           "G4",
           "other"
           }
N_CLASSES = 5
CLASSES_NAMES = ["G1", "G2", "G3", "G4", "other"] #Predicted label: G1:0 G2:1: G3:2 G4:3 other:4
PATCH_SIZE_EXTRACT = 224

#### Set colors for tumor grades ####
norm = mpl.colors.Normalize(vmin=0, vmax=3)
cmap = cm.plasma
m = cm.ScalarMappable(norm=norm, cmap=cmap)
colors = []
for i in range(4):
    colors.append(m.to_rgba(i))
colors.append([0.7, 0.7, 0.7, 1])
mycmap = ListedColormap(colors)

#### Image patch normalization ####
transform_ops = [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                        ),
                    ]
train_transform = transforms.Compose(transform_ops)

#### Load extractor ####
Extractor = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

class MyModel(nn.Module):
    def __init__(self, features_length, classes):
        super(MyModel, self).__init__()
        self.extractor = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        for param in self.extractor.parameters():
            param.requires_grad = False
        
        self.sigmoid1 = nn.Sigmoid()
        self.linear1 = nn.Linear(768, 128, bias=True)
        self.linear2 = nn.Linear(128, classes, bias=True)
        self.sigmoid2 = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        feature = self.extractor(x).last_hidden_state[:, 0, :]
        x = self.sigmoid1(feature)
        x = self.linear1(x)
        x = self.sigmoid2(x)
        out = self.linear2(x)

        return out
 
def extract_feature_from_WSI(input_dir, slide_id,model, out_dir,STEP_SIZE=400,PATCH_SIZE=224,):
    try:
        
        mask = None
        slide = open_slide(os.path.join(input_dir,slide_id))
        magnitude = slide.properties['openslide.objective-power']
        dims = slide.level_dimensions
        
        col_names = ['n_col', 'n_row', 'pred_class']
        col_names.extend(["x{}".format(_) for _ in range(768)])
        slide_df = pd.DataFrame(columns=col_names)
    
        n_rows = int(np.ceil(dims[0][1]/STEP_SIZE))
        n_cols = int(np.ceil(dims[0][0]/STEP_SIZE))
    
    
        if mask is None:
            # Get tissue mask
            mask, _ = get_mask_for_slide_image(os.path.join(input_dir,slide_id))
            mask = skimage.transform.resize(mask, (n_rows, n_cols), order=0, preserve_range=True)
        
        
        for n_row in tqdm(range(int(n_rows))):
            for n_col in range(int(n_cols)):
                
                if mask is None:
                    coord_x = feature_df['coordinate_x']
                    coord_y = feature_df['coordinate_y']
                    coord_x_start = n_col * PATCH_SIZE * 2
                    coord_x_end = (n_col + 1) * PATCH_SIZE * 2
                    coord_y_start = n_row * PATCH_SIZE * 2
                    coord_y_end = (n_row + 1) * PATCH_SIZE * 2
                    expr = "(coord_x >= coord_x_start) & (coord_x < coord_x_end) & (coord_y >= coord_y_start) & (coord_y < coord_y_end)"
                    select = ne.evaluate(expr)

                    if sum(select) == 0:
                        continue

                else:
                    if mask[n_row, n_col] < 1:
                        continue
                
                coord_x_start = n_col * STEP_SIZE
                coord_y_start = n_row * STEP_SIZE
            
                if slide.properties['openslide.objective-power'] == "20":
                    patch = np.array(slide.read_region((coord_x_start, coord_y_start), 0, (int(PATCH_SIZE/2) , int(PATCH_SIZE/2) )), dtype=np.uint8)[..., :3]
                    patch = skimage.transform.resize(patch, (PATCH_SIZE_EXTRACT,PATCH_SIZE_EXTRACT),clip=True,order=1, cval=0, anti_aliasing=True)
                    
                else:
                    patch = np.array(slide.read_region((coord_x_start, coord_y_start), 0, (PATCH_SIZE, PATCH_SIZE)), dtype=np.uint8)[..., :3]

                patch_tensor = train_transform(patch).unsqueeze(0)
                patch_feature = Extractor(patch_tensor).last_hidden_state[:, 0, :]
                patch_feature = patch_feature.squeeze().detach().cpu().numpy()
                out = model(patch_tensor)
                proba = F.softmax(out, dim=1)
                pred = torch.argmax(proba, dim=1).detach().cpu().numpy()[0]

                i = len(slide_df)
                a = np.array([n_col, n_row, pred])
                newRow = np.concatenate((a,patch_feature))
                slide_df.loc[i] = newRow
            
        slide_df[col_names] = slide_df[col_names].apply(np.int64)
        pred_WSI = np.zeros((n_rows, n_cols))
        pred_WSI[:] = np.nan
        for i in slide_df.index.values:
            pred_WSI[slide_df.loc[i, 'n_row'], slide_df.loc[i, 'n_col']] = slide_df.loc[i, 'pred_class']

        # Shinkage border as the border prediction is weird
        mask = np.logical_not(np.isnan(pred_WSI))
        mask = skimage.morphology.erosion(mask, skimage.morphology.disk(2))
        pred_WSI[mask == 0] = np.nan

        #save plot with predict tumor grade
        plt.figure(figsize=(8, 8))    
        plt.imshow(pred_WSI, cmap=mycmap, interpolation="Nearest")
        plt.colorbar()
        plt.savefig(os.path.join(out_dir, "{}_{}cols_{}rows.png".format(slide_id, n_cols, n_rows)))

        # Save data
        slide_df.to_csv(os.path.join(out_dir, "{}_{}cols_{}rows.csv".format(slide_id, n_cols, n_rows)), index=False)
                                     
    except Exception as e:
        print(e)
    

def main(args):
    if args.data_path is not None:
        if os.path.isdir(args.data_path):
            slide_files = [_ for _ in os.listdir(args.data_path) if not _.startswith('.') and (_.endswith(".svs") or _.endswith(".tif") or _.endswith(".tiff") or _.endswith(".dicom") or _.endswith(".ndpi"))]
        else:
            slide_files = [args.data_path]

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load model
    model=MyModel(768,N_CLASSES)
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu') ))
    
    for f in slide_files:
        print(f)
        extract_feature_from_WSI(args.data_path, f, model, args.output_path,STEP_SIZE=400,PATCH_SIZE=224,)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Feature Extraction by phikon')

    parser.add_argument('--data_path', default=os.path.join(os.getcwd(), 'example_data/'), type=str,
                        help="single image file path or a folder path containing image files, support svs, tiff, tif, dicom, and ndpi")
    parser.add_argument('--model', default=os.path.join(os.getcwd(), 'model/'), type=str, help="model")
    parser.add_argument('--pixel_step', default=400, type=int, help='pixel step for patch extraction')
    parser.add_argument('--output_path', default=os.path.join(os.getcwd(), 'output/'), type=str,
                        help='the relative path to output the result')

    args = parser.parse_args()

    main(args)
