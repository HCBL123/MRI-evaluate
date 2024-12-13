import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from torchvision.transforms import RandomRotation as RandomRotate
from torchvision.transforms import RandomAffine
from torchvision.transforms import RandomHorizontalFlip
from dataloader import MRDataset
import model
from sklearn import metrics

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train MRNet Model')
    
    # Required arguments
    parser.add_argument('--task', type=str, required=True,
                      choices=['abnormal', 'acl', 'meniscus'],
                      help='Task to train on')
    parser.add_argument('--plane', type=str, required=True, 
                      choices=['sagittal', 'coronal', 'axial'],
                      help='Plane to train on')
    parser.add_argument('--prefix_name', type=str, required=True,
                      help='Prefix for saved model files')
    
    # Optional arguments
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size') 
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                      help='Patience for early stopping')
    parser.add_argument('--log_every', type=int, default=10,
                      help='Log every N steps')
    
    return parser.parse_args()

def train_model(model, train_loader, epoch, num_epochs, optimizer, writer, current_lr, log_every=100):
    _ = model.train()

    if torch.cuda.is_available():
        model.cuda()

    y_preds = []
    y_trues = []
    losses = []

    for i, (image, label, weight) in enumerate(train_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda() 
            weight = weight.cuda()

        logit = model(image)
        loss = nn.BCEWithLogitsLoss(weight=weight)(logit, label)
        loss.backward()
        optimizer.step()

        y_pred = torch.sigmoid(logit)
        y_preds.extend(y_pred.cpu().detach().numpy())
        y_trues.extend(label.cpu().detach().numpy())
        losses.append(loss.item())

        if (i + 1) % log_every == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, LR: {current_lr}')
            
    train_loss = sum(losses) / len(losses)
    try:
        train_auc = metrics.roc_auc_score(y_trues, y_preds)
    except:
        train_auc = 0.5
        
    return train_loss, train_auc

def main():
    args = parse_arguments()
    
    # Setup logging
    log_dir = os.path.join('logs', args.task, args.plane, 
                          datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Initialize datasets
    train_dataset = MRDataset('./data/', args.task, args.plane, transform=None, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=4)

    # Initialize model and optimizer
    mrnet = model.MRNet()
    if torch.cuda.is_available():
        mrnet = mrnet.cuda()
    
    optimizer = optim.Adam(mrnet.parameters(), lr=args.lr)
    
    # Training loop
    best_val_auc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss, train_auc = train_model(mrnet, train_loader, epoch,
                                          args.epochs, optimizer, writer,
                                          current_lr, args.log_every)
        
        # Early stopping check
        if train_auc > best_val_auc:
            best_val_auc = train_auc
            patience_counter = 0
            # Save model
            save_path = f'models/{args.prefix_name}_{args.task}_{args.plane}_epoch_{epoch+1}.pth'
            torch.save(mrnet.state_dict(), save_path)
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
            
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('AUC/train', train_auc, epoch)
        
    writer.close()

if __name__ == '__main__':
    main()