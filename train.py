import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from src.model import EnhancedTransformerModel
from src.preprocessing import DataPreprocessor
from src.training import CustomDataset, train_model_with_amp, get_optimizer_and_scheduler
from src.utils import compute_metrics

def main():
    # Initialize Data Preprocessor
    preprocessor = DataPreprocessor('allenai/longformer-base-4096', max_length=128)
    texts = ["Example sentence for tokenization.", "Another example sentence."]
    tokenized_texts = preprocessor.tokenize(texts)
    dataset = CustomDataset(texts, preprocessor.tokenizer, max_length=128)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize Model, Optimizer, and Scheduler
    model = EnhancedTransformerModel().to('cuda')
    optimizer, scheduler = get_optimizer_and_scheduler(model, learning_rate=5e-5, train_dataloader=data_loader, num_epochs=3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training
    scaler = GradScaler()
    for epoch in range(3):
        train_loss = train_model_with_amp(model, data_loader, optimizer, criterion, 'cuda', scaler, grad_accumulation_steps=2)
        print(f"Epoch {epoch + 1}, Loss: {train_loss}")

    # Evaluation
    trainer = Trainer(model=model, args=TrainingArguments(output_dir='./results', evaluation_strategy='steps', eval_steps=500), compute_metrics=compute_metrics)
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

if __name__ == '__main__':
    main()
