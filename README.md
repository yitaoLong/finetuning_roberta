# Finetuning RoBERTa
This code is to compare two ways of finetuning RoBERTa model on ag_news dataset. One approach is to finetune the whole model, the other one is only finetuning the last two layers.

## arguments
```
--train: finetune on whole model
--train_last_two: finetune on the last two layers
--eval: evaluate on test set
--saved_dir: path of saving model
--model_dir: path to load model
--learning_rate: learning rate
--num_epochs: number of epochs
--batch_size: batch size to train and evaluate
```

## run python
```
python3 main.py [arguments]
```
