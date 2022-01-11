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


Run this line to start training
```
python train.py \
  --dataset {digits,mnist,cifar10,cifar100} \
  --model {digits_cnn,mnist_cnn,cifar10_cnn,cifar100_cnn} \
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
  --url URL \
  --landscape \
  --samples 
```

For more information about this script
```
python train.py -h
```

### Testing
Run this line to test a trained model
```
python test.py \
  --dataset {digits,mnist,cifar10,cifar100} \
  --model {digits_cnn,mnist_cnn,cifar10_cnn,cifar100_cnn} \
  --batch-size BATCH_SIZE \
  --gpu \ 
  --model-path MODEL_PATH \
  --dashboard \
  --port PORT
  --url URL\
  --landscape \
  --samples 
```

Run this line for more information 
```
python test.py -h 
```

### Landscape
Run this line to see the landscape of the loss function for an specific model
```
python landscape.py \
  --dataset {digits,mnist,cifar10,cifar100} \
  --model {digits_cnn,mnist_cnn,cifar10_cnn,cifar100_cnn} \
  --batch-size BATCH_SIZE \
  --gpu \
  --base-model BASE_MODEL \
  --checkpoints-dir CHECKPOINTS_DIR \
  --xrange XRANGE [XRANGE ...] \
  --yrange YRANGE [YRANGE ...] \
  --N N 
```

Run this line to see more information about this script
```
python landscape.py -h
```

![loss_landscape](samples/loss_landscape.png?raw=true "Loss Landscape")