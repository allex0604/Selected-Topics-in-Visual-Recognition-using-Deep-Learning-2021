install the MMdetection on Windows10 : teaching website
                reference : http://www.4k8k.xyz/article/weixin_41922853/118445620

Data preprocess:
  1. transform data to compatible the model (here is coco format)
  2. put data in data/coco
  data/  
  coco/  
    annotations/  
       instance_train2017.json  
       instance_val2017.json       
       instance_test2017.json       
     train/  
         xxx.png         
     val/
        xxx.png      
     test/  
         xxx.png  


Train and validation:
  must run the following command in the mmdetection folder
  
  python tools/train.py {the position of model.py}





pretrained weight : 
  https://drive.google.com/file/d/1fhpOVfWGyndFDm9V_oxDztO2_SVTnmce/view?usp=sharing
