# pytorch_trainer
Simple script for training computer vision models for different datasets

## How to use
Running the server in case you are using the dashboard
```
python server.py
```

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
  --port PORT 
```

For more information about this script
```
python train.py -h
```