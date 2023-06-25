import os


PROJECT_NAME = 'Glaucoma-Detection'
ENTITY_NAME = None # Wandb username/entityname


SAVE_MODEL_PATH = './saved_models/'


RUN_NAME = 'alternetk128'  # RunType-ModelType-Version
NUM_FOLDS=5 # for cross-validation
DEFAULT_HP = {
            'model':'alternetk',  #'efficientnet', 'resnet', 'resnet50', 'vgg', 'mobilenet','alternetk'
                             #  deits, vit, swintiny, swinbase
                                             
            'model_size': '50',  # resnet:'18', '34', '50' ; effinet 'b0', 'b1';
            'pretrained': False,
            
            'head_dim':16,
            'layers_str':'ResMsa-ResMsa-ResMsa-ResMsa-ResMsa',   #max 5 layers; 'Res', 'ResMsa', 
             
            'in_channels_str':'32-64-128-128-128', #'32-64-128-128-128', '32-64-128-256-256'
            'out_channel_str':'64-128-128-128-128', #'64-128-128-128-128', '64-128-256-256-256'
            'heads_str':'2-4-4-4-4', #'2-4-4-4-4', '2-4-8-8-8'
            
            'lr':0.001,  # 2e-4, 0.001,
            'batch_size':32,
            'beta1': 0.9,
            'beta2': 0.9,
            'weight_decay': 0.0,
            'dropout_rate':0.1,
            'epochs': 150,
            
            #pooling
            'pooling_type':'GAP', # 'GAP', 'LSE'
            'LSE_gamma':None, # 10.0 for LSE pooling,

            #lr schedulers
            'lr_scheduler':'ROP', # 'ROP', 'CAWS'
            'factor':0.1, #reduce on plateau decay factor 
            'patience':10, # reduceOnPlateau
            'T_0':164,  #164 for CAWS
            
            'size':224, # 224

            }

DATASET_PATH = './dataset_224' #path to dir containing images
DATASET_CSV_PATH= './dataset.csv' #path to csv file containing image IDs

ALTERNETK_MODEL_NAMES = ['alternetk',]
HF_MODEL_NAMES = ['deits', 'vit', 'swintiny', 'swinbase']


if DEFAULT_HP["model"] in HF_MODEL_NAMES:
    DEFAULT_HP["hf"]=True
    DEFAULT_HP['pooling_type']=None
    DEFAULT_HP['model_size'] = None
    DEFAULT_HP['lr_scheduler'] = "CAWS"
    
else:
    DEFAULT_HP["hf"]=False


if DEFAULT_HP['model']=='resnet':
    DEFAULT_HP['dropout_rate']=None  
    if DEFAULT_HP['model_size'] not in ('18','34','50'):
        raise RuntimeError() 

if DEFAULT_HP['model']=='efficientnet':
    if DEFAULT_HP['model_size'] not in ('b0','b1'):
        raise RuntimeError() 

    
if DEFAULT_HP['model']=='vgg':
    DEFAULT_HP['pooling_type']=None
    DEFAULT_HP['model_size'] = None
       
       
if DEFAULT_HP['pooling_type']!='LSE':
    DEFAULT_HP['LSE_gamma']=None
 
    
if DEFAULT_HP['model']!="alternetk":
    DEFAULT_HP['head_dim'] = None 
    DEFAULT_HP['layers_str'] = None
    DEFAULT_HP['in_channels_str'] = None
    DEFAULT_HP['out_channel_str'] = None
    DEFAULT_HP['heads_str'] = None
else:
    DEFAULT_HP['lr_scheduler'] = "CAWS"
    DEFAULT_HP['model_size'] = None
    DEFAULT_HP['pooling_type'] = 'GAP'
    
    
if DEFAULT_HP['lr_scheduler']!='ROP':
    DEFAULT_HP['factor'] = None
    DEFAULT_HP['patience'] = None
    
if DEFAULT_HP['lr_scheduler']!='CAWS':
    DEFAULT_HP['T_0'] = None
  
  
CHECKPOINT_MODEL_PATH = os.path.join(SAVE_MODEL_PATH , RUN_NAME + '_CM_.pt') 