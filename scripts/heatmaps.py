import os
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


@torch.no_grad()
def get_cam_max(cam):
    cam = F.relu(cam)
    C,H,W = cam.size()
    cam = cam/(torch.max(cam.view(C,-1), dim=-1, keepdim=True)[0].view(C,1,1) + 1e-9)
    return cam

class HeatmapDataset(torch.utils.data.Dataset):
    def __init__(self, df, path):
        super().__init__()
        self.df = df
        self.path = path
        
        self.length = len(df)
        
        self.transforms = tr.Compose([tr.ToTensor(),
                                      tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                     ])
                
    def __getitem__(self, index):
        label, img_path = self.df.loc[index, 'class'], self.df.loc[index, 'challenge_id']
        img_path = os.path.join(self.path, img_path + '.jpg')
        label = torch.tensor(label).int()
        image = Image.open(img_path).convert('RGB')
        return self.transforms(image), image
        
    def __len__(self):
        return self.length
    
    
# used to generate heatmaps outputs for a model
class Heatmap:
    def __init__(self,model, df, path):
        self.dataset = HeatmapDataset(df, path)
        self.df = df
        self.model = model
        self.device = torch.device('cuda') if torch.cuda.is_available()==True else torch.device('cpu')
        self.model.to(device=self.device)
        self.model.eval()
        self.class_index_to_name = {0:'NRG', 1:'RG'}
        self.compute_results()
        
    def compute_results(self):
        pred_list = None
        for i in range(len(self.dataset)):
            img, _ = self.dataset[i]
            img = img.to(device=self.device)
            pred = self.model(img.unsqueeze(0)).squeeze(0).detach().sigmoid()
            
            pred = (pred>0.5).int()
            
            if pred_list is None:
                pred_list = pred
            
            else:
                pred_list = torch.cat((pred_list, pred), dim=0)    
            
        pred_list = pred_list.cpu().numpy()
        self.df['pred'] = pd.Series(pred_list)
    
    def get_selective(self, class_type):
        """ Gets detailed list falsely predicted image belonging to class class_type 

        Args:
            class_type (_type_): [0,1]

        Returns:
            pd.DataFrame: containing falsely predicted image details
        """
        df = self.df[self.df['class']==class_type]
        df = df[df['class']!=df['pred']]
        return df
    
    def show_heatmap(self, index, figure_scale = 64, alpha=0.6, gamma=0):
        img, ori_img = self.dataset[index]
        img = img.to(device= self.device)
        cam, pred = self.model.predict(img.unsqueeze(0))
        cam = cam.squeeze(0).expand(3,-1,-1).permute(1,2,0).detach()
        cam = get_cam_max(cam).cpu().numpy()
        pred = pred.squeeze()
        pred = (pred>0.5).int().item()
        img = np.uint8(np.array(ori_img))
    
        dx, dy = 0.05, 0.05
        x = np.arange(-3.0, 3.0, dx)
        y = np.arange(-3.0, 3.0, dy)
        extent = np.min(x), np.max(x), np.min(y), np.max(y)
        rows = 1
        columns = 1
        
        fig = plt.figure(figsize=(rows*figure_scale, columns*figure_scale))

        fig.add_subplot(rows,columns,1)
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)###
        print(img.shape, heatmap.shape)
        overlayed_img = cv2.addWeighted(img, alpha, heatmap, 1-alpha,gamma)
        overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)
        plt.imshow(overlayed_img, extent=extent) 
        plt.title(self.class_index_to_name[pred])   
        
        
        
            
        
            