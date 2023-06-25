import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from tqdm import tqdm 
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve


def plot_roc(pred, target, wandb_p, model_version):
    
    num_classes = pred.shape[-1]
    pred, target = pred.cpu().numpy(), target.cpu().numpy()
    line_type = ["-"]
    colour = ['b']

    plt.figure(figsize=(8,8))
    lw = 1

    plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--", label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.title("Receiver operating characteristic curve", fontsize = 15)


    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(target[:, i], pred[:, i])
        plt.plot(
        fpr,
        tpr,
        label='ROC curve',
        color = colour[i],
        linestyle=line_type[i],
        linewidth=1)

    plt.legend(loc="lower right", fontsize=20)
    wandb_p.log({'ROC_CURVE_'+model_version: wandb_p.Image(plt)})


def train_step_function(model, 
               optimizer = None, 
               metrics_fn_dict=None, 
               train_loader = None,
               device = None,
               loss_fn = None,
               grad_scalar = None,
               lr_scheduler = None,
               feature_extractor = None
                ):
    
    if lr_scheduler is None:
        lr_scheduler={}
        
    model.train()
    loss_epoch = 0
    count = 0
    preds_list = None
    targets_list = None
    
    metric_value_dict = {}
    
    for data, target in tqdm(train_loader, desc='Train'):
        if feature_extractor is None:
            data, target = data.to(device = device), target.to(device= device)
            ##### train-step code starts here ######
            
            optimizer.zero_grad(set_to_none=True)
            preds = model(data)
            loss = loss_fn(preds, target.float())
            loss.backward()
            optimizer.step()
            
            ###### train-step code ends here #####
        
        else:
            hf_inputs = feature_extractor(data, return_tensors="pt")
            hf_inputs={k:v.to(device=device) for k,v in hf_inputs.items()}
            target=target.to(device= device)
            ##### train-step code starts here ######
            
            optimizer.zero_grad(set_to_none=True)
            hf_outputs=model(**hf_inputs)
            preds = hf_outputs.logits
            loss = loss_fn(preds, target.float())
            loss.backward()
            optimizer.step()
            
            ###### train-step code ends here #####
            
        if 'CAWS' in lr_scheduler:
            lr_scheduler['CAWS'].step()
        
        loss_epoch+=loss.detach()
        count+=1
        
        if preds_list == None:
            preds_list = preds.detach()
        else:    
            preds_list = torch.concat((preds_list, preds.detach()), dim=0)
            
        if targets_list == None:
            targets_list = target.detach()
        else:    
            targets_list = torch.concat((targets_list, target.detach()), dim=0)
        
    loss_epoch = loss_epoch/count

    if metrics_fn_dict is not None:        
        for key in metrics_fn_dict.keys():        
            metric_value_dict[key+'_train'] = metrics_fn_dict[key](preds_list, targets_list) 
            
    metric_value_dict['Loss'+'_train'] = loss_epoch
        
    return model, metric_value_dict
    

@torch.no_grad()    
def eval_step_function(model,  
               metrics_fn_dict=None, 
               eval_loader = None,
               device = None,
               loss_fn = None,
               metric_value_dict = None,
               feature_extractor = None
               ):
    
    model.eval()
    loss_epoch = 0
    count = 0
    preds_list = None
    targets_list = None
    
    if metric_value_dict is None:
        metric_value_dict = {}
    
    for data, target in tqdm(eval_loader, desc='Eval'):
        if feature_extractor is None:
            data, target = data.to(device = device), target.to(device = device)
            
            ##### eval-step code starts here ######
            
            preds = model(data)
            loss = loss_fn(preds, target.float())
            
            ###### eval-step code ends here #####
        else:
            hf_inputs = feature_extractor(data, return_tensors="pt")
            hf_inputs={k:v.to(device=device) for k,v in hf_inputs.items()}
            target=target.to(device= device)
            
            ##### eval-step code starts here ######
            
            hf_outputs=model(**hf_inputs)
            preds = hf_outputs.logits
            loss = loss_fn(preds, target.float())
            
            ###### eval-step code ends here #####
            
            
        loss_epoch+=loss.detach()
        count+=1
        
        if preds_list == None:
            preds_list = preds.detach()
        else:    
            preds_list = torch.concat((preds_list, preds.detach()), dim=0)
            
        if targets_list == None:
            targets_list = target.detach()
        else:    
            targets_list = torch.concat((targets_list, target.detach()), dim=0)
    
    
    loss_epoch = loss_epoch/count
    
    if metrics_fn_dict is not None:        
        for key in metrics_fn_dict.keys():        
            metric_value_dict[key+'_eval'] = metrics_fn_dict[key](preds_list, targets_list) 
            
    metric_value_dict['Loss'+'_eval'] = loss_epoch
             
        
    return model, metric_value_dict  


def train_model(model, 
               optimizer = None, 
               metrics_fn_dict=None, #{'train_acc':AccuracyObj,}
               train_loader = None,
               eval_loader = None,
               device = None,
               loss_fn = None,
               grad_scalar = None,
               epochs = None,
               wandb_p = None,
               check_point= None, #{'name':str, 'value':float, 'type':'max', 'path':str}
               final_save_path = None,
               lr_scheduler = None,
               feature_extractor=None
               ):  
    
    if lr_scheduler is None:
        lr_scheduler={}
    
    if check_point is not None:
        if check_point['value'] is None:
            if check_point['type']=='max':
                check_point['value']=-1e9
            
            elif check_point['type']=='min': 
                check_point['value']= 1e9 
            
            else:
                raise RuntimeError    
   
    
    if final_save_path is not None:
        model.save_model_weights(final_save_path)
    
    
    for epoch in range(epochs):
        
        print('Epoch', epoch)
        
        model, metric_value_dict = train_step_function(model, 
                                                        optimizer = optimizer, 
                                                        metrics_fn_dict=metrics_fn_dict, 
                                                        train_loader = train_loader,
                                                        device = device,
                                                        loss_fn= loss_fn,
                                                        grad_scalar = grad_scalar,
                                                        lr_scheduler= lr_scheduler,
                                                        feature_extractor=feature_extractor
                                                       )
        
        model, metric_value_dict = eval_step_function(model,  
                                                        metrics_fn_dict= metrics_fn_dict, 
                                                        eval_loader = eval_loader,
                                                        device = device,
                                                        loss_fn= loss_fn,
                                                        metric_value_dict = metric_value_dict,
                                                        feature_extractor=feature_extractor
                                                        )
        
        if 'ROP' in lr_scheduler:
            lr_scheduler['ROP'].step(metric_value_dict['Accuracy_eval'])
        
        # wandb log
        if wandb_p is not None:
            wandb_p.log(metric_value_dict)
        
        #checkpointing
        if check_point is not None:
            if check_point['type'] == 'max':
                if check_point['value']<metric_value_dict[check_point['name']]:
                    check_point['value'] = metric_value_dict[check_point['name']]
                    check_point["metric_values"]=metric_value_dict
                    check_point["epoch"]=epoch
                    torch.save(model, check_point['path']) 
                    if wandb_p is not None:
                        wandb_p.run.summary[check_point['name']+'_Checkpoint'] = check_point['value']
                    
            
            elif check_point['type'] == 'min':
                if check_point['value']>metric_value_dict[check_point['name']]:
                    check_point['value'] = metric_value_dict[check_point['name']]
                    check_point["metric_values"]=metric_value_dict
                    check_point["epoch"]=epoch
                    torch.save(model, check_point['path']) 
                    if wandb_p is not None:
                        wandb_p.run.summary[check_point['name']+'_Checkpoint'] = check_point['value']
                        
            else:
                raise RuntimeError    
              
    if final_save_path is not None:
        model.save_model_weights(final_save_path)                     
   
    if check_point is not None:
        return model, check_point   

    else:
        return model


@torch.no_grad()
def test_model(model,  
               wandb_p,
               metrics_fn_dict=None, 
               test_loader = None,
               device = None,
               loss_fn = None,
               model_version = None,
               ):
    
    model.eval()
    loss_epoch = 0
    count = 0
    preds_list = None
    targets_list = None
    
    
    for data, target in tqdm(test_loader, desc='Test'):
        data, target = data.to(device = device), target.to(device = device)
        
        ##### test-step code starts here ######
        
        preds = model(data)
        loss = loss_fn(preds, target.float())
        
        ###### test-step code ends here #####
            
        loss_epoch+=loss.detach()
        count+=1
        
        if preds_list == None:
            preds_list = preds.detach()
        else:    
            preds_list = torch.concat((preds_list, preds.detach()), dim=0)
            
        if targets_list == None:
            targets_list = target.detach()
        else:    
            targets_list = torch.concat((targets_list, target.detach()), dim=0)
    
    
    loss_epoch = loss_epoch/count
    
    if metrics_fn_dict is not None:         
        for key in metrics_fn_dict.keys():        
            wandb_p.run.summary[key+'_test_'+ model_version] = metrics_fn_dict[key](preds_list, targets_list) 
            
    wandb_p.run.summary['Loss'+'_test_'+ model_version] = loss_epoch
    # print('preds_list.shape:',preds_list.shape)
    preds_class1 = preds_list.unsqueeze(-1)
    preds_class0 = 1 - preds_class1
    preds_roc_list = torch.concat((preds_class0, preds_class1), dim = 1)
    
    preds_cm = (preds_list>0.5).int()
    
    plot_roc(preds_list, targets_list, wandb_p, model_version)
    # wandb_p.log({'ROC_'+ model_version: wandb_p.plot.roc_curve(targets_list.cpu(), preds_roc_list.cpu(), labels =['NRG', 'RG'])})  
    if  model_version=='Checkpoint':
        wandb_p.sklearn.plot_confusion_matrix(targets_list.cpu(),preds_cm.cpu(), ['NRG', 'RG'])
    return None

