from datasets import load_dataset
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification
from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import argparse
from tqdm.auto import tqdm


def tokenizer_function(examples):
    '''
    Tokenize each sample
    '''
    return tokenizer(examples['text'], padding='max_length', truncation=True)

def training(args, model, train_dataloader):
#    define optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
#    define learning rate scheduler
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
#    set model to training mode
    model.train()
    progress_bar = tqdm(range(num_training_steps))

#   define a loss list to record loss in each step
    loss = []
    for _ in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            optimizer.zero_grad()
            loss.append(outputs.loss.item())
            outputs.loss.backward()
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)
    
#    save model
    model.save_pretrained(args.saved_dir)
#    write loss to a file
    if args.train:
        with open('whole_model_loss.txt', 'w') as f:
            f.write(' '.join([str(d) for d in loss]))
    else:
        with open('two_layer_loss.txt', 'w') as f:
            f.write(' '.join([str(d) for d in loss]))
    print('Training finished!')


def evaluating(args, model, test_dataloader):
#   define TP, TN, FP, FN to compute precsion, recall, F1 score of each label
    TP = [0 for _ in range(4)]
    TN = [0 for _ in range(4)]
    FP = [0 for _ in range(4)]
    FN = [0 for _ in range(4)]

#   set model to evaluate model
    model.eval()

    for i, batch in enumerate(tqdm(test_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)

        for l in range(4):
            TP[l] += sum([True if predictions[j] == l and batch['labels'][j] == l else False for j in range(len(predictions))])
            TN[l] += sum([True if predictions[j] != l and batch['labels'][j] != l else False for j in range(len(predictions))])
            FP[l] += sum([True if predictions[j] == l and batch['labels'][j] != l else False for j in range(len(predictions))])
            FN[l] += sum([True if predictions[j] != l and batch['labels'][j] == l else False for j in range(len(predictions))])

    print('Accuracy: ' + str(sum(TP)/7600))
    for i in range(4):
        if TP[i] + FP[i] > 0:
            precison = TP[i] / (TP[i] + FP[i])
        else:
            precison = 0
        if TP[i] + FN[i] > 0:
            recall = TP[i] / (TP[i] + FN[i])
        else:
            recall = 0
        if precison + recall > 0:
            f1score = 2 * precison * recall / (precison + recall)
        else:
            f1score = 0
        print('Precision of label ' + str(i) + ': ' + str(precison))
        print('Recall of label ' + str(i) + ': ' + str(recall))
        print('F1 score of label ' + str(i) + ': ' + str(f1score))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # arguments
    parser.add_argument("--train", action="store_true", help="finetuning the whole model")
    parser.add_argument("--train_last_two", action="store_true", help="finetuning only the last two layers")
    parser.add_argument("--eval", action="store_true", help="evaluate model")
    parser.add_argument("--saved_dir", type=str, default="./out")
    parser.add_argument("--model_dir", type=str, default="./out")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    
    args = parser.parse_args()

    global device
    global tokenizer

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#   load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
#   load dataset
    dataset = load_dataset('ag_news')
#   preprocess on dataset
    tokenized_dataset = dataset.map(tokenizer_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=args.batch_size)
    test_dataloader = DataLoader(tokenized_dataset['test'], batch_size=args.batch_size)

    if args.train:
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=4)
        model.to(device)
        training(args, model, train_dataloader)

    if args.train_last_two:
#    since we are going to finetuning the last two layers, freeze the parameters of embedding layer and the first 10 encoder blocks
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=4)
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = False
        for i in range(10):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False
        model.to(device)
        training(args, model, train_dataloader)
      
    if args.eval:
        model = RobertaForSequenceClassification.from_pretrained(args.model_dir)
        model.to(device)
        evaluating(args, model, test_dataloader)
