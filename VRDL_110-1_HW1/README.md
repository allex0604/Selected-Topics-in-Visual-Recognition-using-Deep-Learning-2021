model download: 
  1. pip install geffnet 
  2. directly use torch.hub.list('rwightman/gen-efficientnet-pytorch')
 

test.py is used to check the accuracy and overfitting , the code can not create answer

hw1.py is used to train the model with all training data ,and get the pretrained_weight called : "my_model.pth" to predict test data and create answer
