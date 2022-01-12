reference :
                      
    https://github.com/Paper99/SRFBN_CVPR19

generate training pair: use matlab to run generate_train_data.m


train :

1.    Edit SRFBN_CVPR19/options/train/train_SRFBN.json for your needs according to SRFBN_CVPR19/options/train/README.md

2.        python train.py -opt options/train/train_SRFBN.json

test :


1.    only need to change the pretrained weight position on  SRFBN_CVPR19/options/test/test_SRFBN.json
2.        python test.py -opt options/test/test_SRFBN.json
3.    you can see the result on SRFBN_CVPR19/results/SR/MyImage



pretrained weight:

    https://drive.google.com/file/d/1ZdzgYGQF1X3lz_Och7EyGhOuXAklNwsh/view?usp=sharing
