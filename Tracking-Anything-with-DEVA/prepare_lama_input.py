from PIL import Image
import numpy as np
import os
import shutil
import cv2
import sys

def mask_to_bbox(mask):
    # Find the rows and columns where the mask is non-zero
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    return xmin, ymin, xmax, ymax

# Check if the user provided an argument
# if len(sys.argv) != 4:
if len(sys.argv) < 4:
    print("Usage: python3 {} <img_path> <mask_path> <lama_path>".format(sys.argv[0]))
    sys.exit(1)

dataset_name = sys.argv[1]

image_dir = sys.argv[1]
mask_dir = os.path.join(sys.argv[2],"Annotations")
out_dir = sys.argv[3]
out_mask_dir = os.path.join(sys.argv[3],"label")
out_mask_vis_dir = os.path.join(sys.argv[3],"label_vis")
# test_dir = os.path.join(sys.argv[3],"test")
os.makedirs(out_dir,exist_ok=True)
os.makedirs(out_mask_dir,exist_ok=True)
os.makedirs(out_mask_vis_dir,exist_ok=True)
# os.makedirs(test_dir,exist_ok=True)





print("Image dir:   ", image_dir)
print("Mask dir:   ", mask_dir)
print("Lama input dir:   ", out_dir)

if len(sys.argv) == 5:   
    object_mask_dir = sys.argv[4] #../data/teatime/inpaint_object_mask_255
    dirs = []
    names = []
    for dir in os.listdir(object_mask_dir):
        sub_dir = os.path.join(object_mask_dir,dir)
        if(os.path.isdir(sub_dir)):
            dirs.append(sub_dir)
            names.append(sorted(os.listdir(sub_dir)))

for i,name in enumerate(sorted(os.listdir(image_dir))):

    print(os.path.join(image_dir,name))
    shutil.copy(os.path.join(image_dir,name),os.path.join(out_dir,name))

    mask = cv2.imread(os.path.join(mask_dir,name))

    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY_INV)
    binary_mask = cv2.bitwise_not(binary_mask)

    if len(sys.argv) == 5:
    #generate bounding mask here
        bounding_mask = np.zeros_like(gray_mask,dtype=bool)
        for sub_index,sub_dir in enumerate(dirs):
            sub_bounding_mask = cv2.imread(os.path.join(sub_dir,names[sub_index][i]), cv2.IMREAD_GRAYSCALE)
            if np.all(np.equal(sub_bounding_mask,0)):
                continue
            # bounding = mask_to_bbox(sub_bounding_mask)
            # bounding_mask[bounding[1]:bounding[3],bounding[0]:bounding[2]] = True
            # bounding_mask_save = bounding_mask.astype(np.uint8) * 255
            _, sub_bounding_mask = cv2.threshold(sub_bounding_mask, 128, 255, cv2.THRESH_BINARY_INV)
            # cv2.imwrite(os.path.join("../test",f"{i}.png"), sub_bounding_mask)
            # non_binary_indices = np.where((sub_bounding_mask != 0) & (sub_bounding_mask != 255))
            sub_bounding_mask = cv2.bitwise_not(sub_bounding_mask)
            bounding_mask[sub_bounding_mask!=0] = True
        
        bounding_mask = bounding_mask.astype(np.uint8) * 255
        # cv2.imwrite(os.path.join("../test",f"{i}.jpg"), bounding_mask)
        binary_mask = binary_mask & bounding_mask

    # You can change the mask dilation kernel size and dilated_iterations according to your dataset.
    kernel_size = 5 
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_iterations = 5
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=dilated_iterations)  

    cv2.imwrite(os.path.join(out_mask_vis_dir,name), dilated_mask)
    dilated_mask[dilated_mask>0] = 1
    cv2.imwrite(os.path.join(out_mask_dir,name), dilated_mask)



