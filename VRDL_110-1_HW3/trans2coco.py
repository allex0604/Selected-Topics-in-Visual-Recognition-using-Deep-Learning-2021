from numpy import append
from create_annotations import *

# Label ids of the dataset
category_ids = {
    "background" : 0,
    "Nuclei": 1
}

# Define which colors match which categories in the images
category_colors = {
    "(0, 0, 0)" : 0, # background
    "(255, 255, 255)": 1 # Nuclei

}


# Get "images" and "annotations" info 
def images_annotations_info(train_path, maskpath, img_name, image_id):
    # This id will be automatically increased as we go
    annotations = []
    img_name = img_name+".png"
    images = []
    annotation_id = 0
    #training image info
    img_path =os.path.join(train_path, img_name)
    # Open the image and (to be sure) we convert it to RGB
    train_image_open = Image.open(img_path).convert("RGB")
    w, h = train_image_open.size
    image = create_image_annotation(img_name, w, h, image_id)

    for mask_image in os.listdir(maskpath):
        print(mask_image)
        images.append(image)
        img_path =os.path.join(maskpath, mask_image)
        # Open the image and (to be sure) we convert it to RGB
        mask_image_open = Image.open(img_path).convert("RGB")
        w, h = mask_image_open.size
        sub_masks = create_sub_masks(mask_image_open, w, h)
        for color, sub_mask in sub_masks.items():
            category_id = category_colors[color]
            if category_id==0:
                continue
            # "annotations" info
            polygons, _ = create_sub_mask_annotation(sub_mask)
            for i in range(len(polygons)):
                # Cleaner to recalculate this variable
                segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                    
                annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)
                    
                annotations.append(annotation)
                annotation_id += 1
    return images, annotations, annotation_id

if __name__ == "__main__":
    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()
    p = os.path.join("dataset", "train")
    image_id = 0
    for img_name in os.listdir(p):
        print(img_name)
        if image_id<23:
            image_id += 1
            continue
        train_path = os.path .join(p, img_name, "images")
        mask_path = os.path.join(p,img_name,"masks")
    
        # Create images and annotations sections
        coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(train_path, mask_path, img_name, image_id)
        with open("{}.json".format(img_name),"w") as outfile:
            json.dump(coco_format, outfile)
        print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_path))
        image_id += 1


