import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup

class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in item.items()}

def train_model_with_amp(model, data_loader, optimizer, criterion, device, scaler, grad_accumulation_steps=1):
    model.train()
    total_loss = 0
    for i, batch in enumerate(data_loader):
        inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, model.classifier.out_features), labels.view(-1))
        scaler.scale(loss).backward()
        if (i + 1) % grad_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def get_optimizer_and_scheduler(model, learning_rate, train_dataloader, num_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    return optimizer, scheduler
