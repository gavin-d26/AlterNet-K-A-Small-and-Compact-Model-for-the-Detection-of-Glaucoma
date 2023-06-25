import os
import PIL.Image as Image

# This script is used to reduce the Aspect Ratio of images

SRC_DIR = r'./dataset'

DES_DIR_224 = r'./dataset_224'

AR = 1.1 # desired Aspect Ratio

count=0
print(SRC_DIR)

for img_id in os.listdir(SRC_DIR):
    src_img_path = os.path.join(SRC_DIR, img_id)
    
    des_img_path_224 = os.path.join(DES_DIR_224, img_id)
    
    img = Image.open(src_img_path)
    img = img.convert('RGB')
    
    width, height = img.size
    
    left = (width-AR*height)//2
    top = 0
    right = width - left
    bottom = height
    
    img = img.crop((left, top, right, bottom))
    
    img_224 = img.resize((224,224), resample= Image.BILINEAR)
    
    img_224.save(des_img_path_224)
    
    count+=1

print(count)    