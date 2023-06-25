import numpy as np 
import os
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
import torchmetrics as tm
import random
import wandb
import models
import config
from utils import *
import dataset
import std_models


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

"""
learning_rate:
beta1:
beta2:
batch_size:
weight_decay:
learning_rate_decay:
patience: 6

"""
wandb_config_defaults = config.DEFAULT_HP

PROJECT_NAME = config.PROJECT_NAME
ENTITY_NAME = config.ENTITY_NAME
RUN_NAME = config.RUN_NAME
CHECKPOINT_MODEL_PATH = config.CHECKPOINT_MODEL_PATH
DATASET_PATH = config.DATASET_PATH
DATASET_CSV_PATH = config.DATASET_CSV_PATH
NUM_FOLDS=config.NUM_FOLDS

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__=="__main__":
    dataset_df = pd.read_csv(DATASET_CSV_PATH, index_col=0)
    folds_metric_list=[]
    
    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)
    for fold, (train_index, test_index) in enumerate(kf.split(dataset_df["challenge_id"], dataset_df["class"])):
        print(f"Fold: {fold}")
        
        checkpoint_model_path=CHECKPOINT_MODEL_PATH.replace(".pt", "")
        if os.path.isdir(checkpoint_model_path) is False:
            os.makedirs(checkpoint_model_path)
        checkpoint_model_path=os.path.join(checkpoint_model_path, checkpoint_model_path.split("/")[-1]+"Fold-"+str(fold)+'.pt')
        
        train_df, val_df = dataset_df.iloc[train_index].reset_index().drop(columns=["index"]), dataset_df.iloc[test_index].reset_index().drop(columns=["index"])
    
        train_dataset = dataset.EyeDataset(train_df,
                                        DATASET_PATH, 
                                        hf=wandb_config_defaults['hf']
                                        )
        
        validation_dataset = dataset.EyeDataset(val_df, 
                                            DATASET_PATH, 
                                            validation = True,
                                            hf=wandb_config_defaults['hf']
                                            )

    
        # train_dataset = torch.utils.data.Subset(train_dataset, range(32)) ###
        # validation_dataset = torch.utils.data.Subset(validation_dataset, range(32))  ###
        
        
        wandb.init(project = PROJECT_NAME, entity = ENTITY_NAME, group = RUN_NAME, job_type="Fold-"+str(fold), config = wandb_config_defaults)
        wandb_config = wandb.config   
        
    
        print(RUN_NAME+"_"+"Fold-"+str(fold))
        print(DATASET_PATH)
    
                                                    
        collate_fn=dataset.hf_collate if wandb_config_defaults['hf'] is True else None                                            
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size = wandb_config.batch_size, 
                                                pin_memory = True,
                                                shuffle = True,
                                                drop_last = False,
                                                collate_fn=collate_fn,
                                                num_workers = 2)
        
        validation_loader = torch.utils.data.DataLoader(validation_dataset, 
                                                        batch_size = wandb_config.batch_size, 
                                                        pin_memory = True,
                                                        shuffle = False,
                                                        drop_last = False,
                                                        collate_fn=collate_fn,
                                                        num_workers = 2)
        
        
        #### ALL CHANGES BELOW ####
        
        metrics_fn_dict = {'Accuracy':tm.Accuracy().to(device = DEVICE), 
                        'AUROC':tm.AUROC().to(device = DEVICE), 
                        'Precision':tm.Precision().to(device = DEVICE), 
                        'Recall':tm.Recall().to(device = DEVICE), 
                        'F1_score':tm.F1Score().to(device = DEVICE),
                        'Specificity':tm.Specificity().to(device=DEVICE)}
        
        feature_extractor=None
        
        loss_fn = torch.nn.BCEWithLogitsLoss()
        check_point = {'name':'Accuracy_eval','value':None, 'path':checkpoint_model_path, 'type':'max'}####
        
        #model initialization
        if wandb_config.model == 'resnet':
            model = std_models.define_resnet(wandb_config.pooling_type, 
                                            wandb_config.model_size,
                                            pretrained=wandb_config.pretrained, 
                                            gamma=wandb_config.LSE_gamma)

            print('USING RESNET')
            
        
            
        elif wandb_config.model == 'efficientnet':
            model = std_models.define_efficientnet(wandb_config.pooling_type, 
                                                    wandb_config.model_size,
                                                    gamma = wandb_config.LSE_gamma, 
                                                    dropout=wandb_config.dropout_rate, 
                                                    pretrained = wandb_config.pretrained) 
            print('USING EFFICIENTNET')

            
        elif wandb_config.model == 'vgg':
            model = std_models.define_vgg(dropout=wandb_config.dropout_rate,
                            pretrained=wandb_config.pretrained)  
            print('USING VGG')
    
        
        elif wandb_config.model == 'mobilenet':
            model = std_models.define_mobilenet(wandb_config.pooling_type,
                            gamma = wandb_config.LSE_gamma,
                            dropout= wandb_config.dropout_rate,
                            pretrained= wandb_config.pretrained)
            print('USING MOBILENET')
        
        
        elif wandb_config.model == 'deits':
            model, feature_extractor = std_models.define_deits() 
            
        
        elif wandb_config.model == 'vit':
            model, feature_extractor = std_models.define_vit()
            
            
        elif wandb_config.model == 'swintiny':
            model, feature_extractor = std_models.define_swintiny()
            
            
        elif wandb_config.model == 'swinbase':
            model, feature_extractor = std_models.define_swinbase()


        elif wandb_config.model == 'alternetk':
            model = models.AlterNetK(wandb_config.layers_str, 
                                        wandb_config.in_channels_str, 
                                        wandb_config.out_channel_str, 
                                        wandb_config.heads_str, 
                                        wandb_config.head_dim,
                                        dropout=wandb_config.dropout_rate)
            print('using AlterNetK')
        
            
        # save model parameter count
        num_params = torch.nn.utils.parameters_to_vector(model.parameters()).numel()
        wandb.run.summary['num_params'] = num_params
        
        model.to(device = DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=wandb_config.lr, 
                                    betas=(wandb_config.beta1, wandb_config.beta2),
                                    weight_decay = wandb_config.weight_decay)

        
        lr_scheduler = {}
        
        if 'ROP' in wandb_config.lr_scheduler:
            lr_scheduler['ROP'] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                        patience=wandb_config.patience, 
                                                                        min_lr = 1e-7,
                                                                        mode='max', 
                                                                        verbose=True)
            print('USING ROP LRS')
        
        if 'CAWS' in wandb_config.lr_scheduler:
            lr_scheduler['CAWS'] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                                        T_0 =wandb_config.T_0, 
                                                                                        T_mult=1, 
                                                                                        eta_min=0, 
                                                                                        last_epoch=- 1, 
                                                                                        verbose=False)
            print('USING CAWS LRS')
    
        model, check_point = train_model(model, 
                            optimizer = optimizer, 
                            metrics_fn_dict=metrics_fn_dict, #{'train_acc':AccuracyObj,}
                            train_loader = train_loader,
                            eval_loader = validation_loader,
                            device = DEVICE,
                            loss_fn = loss_fn,
                            epochs = wandb_config.epochs,
                            wandb_p = wandb,
                            check_point= check_point, #{'name':str, 'value':float, 'type':'max', 'path':str}
                            lr_scheduler = lr_scheduler,
                            feature_extractor=feature_extractor)
                            
    
        folds_metric_list.append(check_point["metric_values"])
        wandb.run.summary["epoch_Checkpoint"]=check_point["epoch"]
        for k,v in check_point["metric_values"].items():
            if '_eval' in k:
                wandb.run.summary[k+"_Checkpoint"]=v
            
        
        if fold+1==NUM_FOLDS:
            k_fold_metric_df=pd.DataFrame(folds_metric_list)
            k_fold_metric_mean_dict=k_fold_metric_df.mean().to_dict()
            k_fold_metric_std_dict=k_fold_metric_df.std(ddof=0).to_dict()
            for k in k_fold_metric_df.keys():
                if '_eval' in k:
                    wandb.run.summary[k+"_Mean"]=k_fold_metric_mean_dict[k]
                    wandb.run.summary[k+"_Std"]=k_fold_metric_std_dict[k]
        
        wandb.finish()
    
    