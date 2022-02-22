# pytorch_trainer
Simple script for training computer vision models for different datasets

## How to use
### Training
Running the server in case you are using the dashboard
```
python server.py
```
In the dashboard you can see the evolution of the training of your network, predictions on some samples images, the loss landscape around the optimal parameters found in the training.
![dashboard](samples/dashboard.png?raw=true "Dashboard")
![samples](samples/samples.png?raw=true "Samples")
![landscape](samples/loss_landscape.png?raw=true "Landscape")
![contour](samples/contour.png?raw=true "Contour")

You can also use the following command line using ngrok if you want your dashboard to be available outside your local network. (If you want to use it from google colab for example)
```
ngrok http PORT
```

Run this line to start training
```
python train.py \
  --dataset {digits,mnist,cifar10,cifar100,image_folder} \
  --root ROOT \
  --num-classes NUM_CLASSES \
  --model {digits_cnn,mnist_cnn,cifar10_cnn,cifar100_cnn, simple_general_cnn} \
  --batch-size BATCH_SIZE \
  --gpu  \      
  --optim OPTIM  \        
  --lr LR  \     
  --epochs EPOCHS \
  --logging \      
  --no-logging \       
  --save-model \
  --csv \     
  --dashboard \       
  --port PORT \
  --checkpoint CHECKPOINT \
  --url URL \
  --landscape \
  --samples \
  --start-model START_MODEL_PATH \
  --save-labels \
  --customize \
  --sched {none,step} \
  --step-size 10 \
  --gamma 0.1
```

#### Examples
Using an image folder containing all images in the following format. (Internally the [ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) dataset from torchvision is used)
```
root/
  class1/
    images ...
  class2/
    images ...
  ...
```
Example on how to train
```
python train.py --dataset image_folder --root PATH_TO_ROOT --model simple_general_cnn --num-classes NUMBER_OF_CLASSES --save-model --epochs 10
```

Using a customize model and dataloader. Create a customize.py file (using the following template) in the root folder of this repo.

```python
def create_model_customize(args):
  # ...
  # your code
  return model

def create_dataloader_customize(args):
  # ...
  # your code
  # train_dl: Train dataloader
  # test_dl: Test dataloader
  # test_ds: Test dataset
  # raw_test_ds: Raw test dataset (without any transformation, used for the samples)
  # extra_info: dictonary of extra information ({"image_shape: (3,112,112), "labels": ["LABEL1","LABEL2",...]})
  return train_dl, test_dl, test_ds, raw_test_ds, extra_info
```
Example on how to train
```
python train.py --customize --epochs 10 --save-model
```

For more information about this script
```
python train.py -h
```

### Testing
Run this line to test a trained model
```
python test.py \
  --dataset {digits,mnist,cifar10,cifar100,image_folder} \
  --model {digits_cnn,mnist_cnn,cifar10_cnn,cifar100_cnn,simple_general_cnn} \
  --root ROOT \
  --num-classes NUM_CLASSES \
  --batch-size BATCH_SIZE \
  --gpu \ 
  --model-path MODEL_PATH \
  --dashboard \
  --port PORT \
  --url URL\
  --landscape \
  --samples \
  --customize
```

Run this line for more information 
```
python test.py -h 
```

### Landscape
Run this line to see the landscape of the loss function for an specific model
```
python landscape.py \
  --dataset {digits,mnist,cifar10,cifar100,image_folder} \
  --root ROOT \
  --num-classes NUM_CLASSES \
  --model {digits_cnn,mnist_cnn,cifar10_cnn,cifar100_cnn, simple_general_cnn} \
  --batch-size BATCH_SIZE \
  --gpu \
  --base-model BASE_MODEL \
  --checkpoints-dir CHECKPOINTS_DIR \
  --xrange XRANGE [XRANGE ...] \
  --yrange YRANGE [YRANGE ...] \
  --N N \
  --customize
```

Run this line to see more information about this script
```
python landscape.py -h
```

![loss_landscape](samples/loss_landscape.png?raw=true "Loss Landscape")
