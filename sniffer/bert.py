from transformers import BertTokenizer, BertModel
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

"""
Modified Code from @marcellusruben
- https://github.com/marcellusruben/medium-resources/blob/main/Text_Classification_BERT/bert_medium.ipynb
"""

class Dataset(torch.utils.data.Dataset):
    """
    Custom Torch class for input text data
    """
    def __init__(self, df, tokenizer):
        self.labels = [label for label in df['label']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(nn.Module):
    """
    BERT encoder model with custom feed forward architecture.
    """
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(768, 512)
        self.dense2 = nn.Linear(512, 512)
        self.output = nn.Linear(512, 6)
        self.relu2 = nn.ReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Softmax(dim=-1)

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        x = self.dense1(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.output(x)
        final_layer = self.sigmoid(x)
        return final_layer

def bert_train(model, train_data, val_data, tokenizer):
    """
    Training function for the BERT classifier
    
    Args
    ---------
        model: BERT classifier model
        train_data: pandas df
        val_data: pandas df
        tokenizer: BERT tokenizer
    """

    EPOCHS = 10  
    LR = 1e-3
    BATCH_SIZE = 1024

    train, val = Dataset(train_data, tokenizer), Dataset(val_data, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}')
    
    torch.save(model.state_dict(), 'bert_model.pth')

def bert_evaluate(model, test_data, tokenizer):
    """
    Evaluates the performance of BERT on the test set
    
    Args
    ---------
        model: BERT classifier model
        test_data: pandas df
        tokenizer: BERT tokenizer
    """

    test = Dataset(test_data, tokenizer)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    total_acc_test = 0  
    loss = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              batch_loss = criterion(output, test_label.long())
              loss += batch_loss.item()

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc              
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .4f}')
    print(f'Test CCE Loss: {loss / len(test_data): .4f}')