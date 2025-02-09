import os
import torch
import pandas as pd
import numpy as np
from torch import nn
from transformers import RobertaModel, RobertaTokenizer, ViTModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# Configuration
CONFIG = {
    "text_model": "roberta-base",
    "image_model": "google/vit-base-patch16-224-in21k",
    "max_seq_length": 128,
    "image_size": 224,
    "batch_size": 32,
    "num_epochs": 10,
    "fusion_dim": 512,
    "num_sentiment_classes": 3,
    "num_humor_classes": 3,  # [humor, sarcasm, offensive]
    "num_scales": 4  # [humor, sarcasm, offensive, motivation]
}

class MemeDataset(Dataset):
    def __init__(self, df, text_tokenizer, image_folder):
        self.df = df
        self.tokenizer = text_tokenizer
        self.image_folder = image_folder
        self.transform = transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Text processing
        text = str(row['text'])
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=CONFIG['max_seq_length'],
            return_tensors='pt'
        )
        
        # Image processing
        img_path = os.path.join(self.image_folder, row['image_name'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Labels
        labels = {
            'sentiment': torch.tensor(row['sentiment']),
            'humor': torch.tensor([
                row['humor_label'],
                row['sarcasm_label'],
                row['offense_label']
            ]),
            'scales': torch.tensor([
                row['humor_scale'],
                row['sarcasm_scale'],
                row['offense_scale'],
                row['motivation_scale']
            ])
        }
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'image': image,
            'labels': labels
        }

class MultimodalMemeAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Text Encoder
        self.text_encoder = RobertaModel.from_pretrained(CONFIG['text_model'])
        self.text_proj = nn.Linear(768, 256)
        
        # Image Encoder
        self.img_encoder = ViTModel.from_pretrained(CONFIG['image_model'])
        self.img_proj = nn.Linear(768, 256)
        
        # Multimodal Fusion
        self.cross_attn = nn.MultiheadAttention(256, 4)
        self.fusion_gate = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # Task Heads
        self.sentiment_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, CONFIG['num_sentiment_classes'])
        )
        
        self.humor_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, CONFIG['num_humor_classes'])
        )
        
        self.scale_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, CONFIG['num_scales'])
        )

    def forward(self, text_input, image_input):
        # Text features
        text_output = self.text_encoder(
            input_ids=text_input['input_ids'],
            attention_mask=text_input['attention_mask']
        ).last_hidden_state[:,0,:]
        text_proj = self.text_proj(text_output)
        
        # Image features
        img_output = self.img_encoder(image_input).last_hidden_state[:,0,:]
        img_proj = self.img_proj(img_output)
        
        # Cross-modal attention
        attn_output, _ = self.cross_attn(
            img_proj.unsqueeze(0), 
            text_proj.unsqueeze(0), 
            text_proj.unsqueeze(0)
        )
        
        # Gated fusion
        combined = torch.cat([text_proj, img_proj], dim=-1)
        gate = self.fusion_gate(combined)
        fused_features = gate * text_proj + (1 - gate) * img_proj
        
        # Task outputs
        sentiment = self.sentiment_head(fused_features)
        humor = self.humor_head(fused_features)
        scales = self.scale_head(fused_features)
        
        return {
            'sentiment': sentiment,
            'humor': humor,
            'scales': scales
        }

# Training Utilities
class MultitaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, outputs, labels):
        # Sentiment (CrossEntropy)
        loss_sentiment = self.ce(outputs['sentiment'], labels['sentiment'])
        
        # Humor/Sarcasm/Offense (Multi-label BCE)
        loss_humor = self.bce(outputs['humor'], labels['humor'].float())
        
        # Scales (Ordinal Regression)
        loss_scales = self.mse(outputs['scales'], labels['scales'].float())
        
        return loss_sentiment + loss_humor + loss_scales

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = {'sentiment': [], 'humor': [], 'scales': []}
    all_labels = {'sentiment': [], 'humor': [], 'scales': []}
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = {k: v.to(device) for k, v in batch['labels'].items()}
            
            outputs = model(inputs['input_ids'], inputs['image'])
            
            # Store predictions and labels
            for task in ['sentiment', 'humor', 'scales']:
                all_preds[task].append(outputs[task].cpu())
                all_labels[task].append(labels[task].cpu())
    
    # Calculate metrics
    metrics = {}
    for task in ['sentiment', 'humor', 'scales']:
        preds = torch.cat(all_preds[task])
        labels = torch.cat(all_labels[task])
        
        if task == 'sentiment':
            metrics[f'{task}_f1'] = f1_score(
                labels.numpy(), 
                preds.argmax(dim=1).numpy(), 
                average='macro'
            )
        elif task == 'humor':
            # Calculate F1 for each subtask
            for i, subtask in enumerate(['humor', 'sarcasm', 'offense']):
                metrics[f'{subtask}_f1'] = f1_score(
                    labels[:,i].numpy(), 
                    (preds[:,i] > 0).float().numpy(), 
                    average='macro'
                )
        elif task == 'scales':
            # MSE for regression
            metrics['scale_mse'] = torch.mean((preds - labels.float())**2).item()
    
    return metrics

# Training Loop
def train_model(model, train_loader, val_loader, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = MultitaskLoss()
    
    best_f1 = 0
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = {k: v.to(device) for k, v in batch['labels'].items()}
            
            outputs = model(inputs['input_ids'], inputs['image'])
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        print(f"Train Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"Validation Metrics: {val_metrics}")
        
        # Save best model
        if val_metrics['sentiment_f1'] > best_f1:
            best_f1 = val_metrics['sentiment_f1']
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model

# Usage Example
if __name__ == "__main__":
    # Initialize components
    tokenizer = RobertaTokenizer.from_pretrained(CONFIG['text_model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    df = pd.read_csv('labels.csv')
    train_df, val_df = train_test_split(df, test_size=0.2)
    
    train_dataset = MemeDataset(train_df, tokenizer, 'C:/Users/himan/Downloads/Multimodal_dataset_assignment3/images')
    val_dataset = MemeDataset(val_df, tokenizer, 'C:/Users/himan/Downloads/Multimodal_dataset_assignment3/images')
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    # Initialize model
    model = MultimodalMemeAnalyzer().to(device)
    
    # Train
    trained_model = train_model(model, train_loader, val_loader, device)
