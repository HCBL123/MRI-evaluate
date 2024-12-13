import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
from pathlib import Path

from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

class MRDataset(Dataset):
    def __init__(self, root_dir, task, plane, transform=None, train=True, normalize=True):
        super().__init__()
        self.root_dir = root_dir
        self.task = task
        self.plane = plane
        self.transform = transform
        self.train = train
        self.normalize = normalize
        
        # Set paths based on train/validation
        self.folder_path = os.path.join(self.root_dir, 'train' if self.train else 'valid', plane)
        
        # Load labels
        label_path = os.path.join(self.root_dir, f'{"train" if self.train else "valid"}-{task}.csv')
        self.records = pd.read_csv(label_path, header=None, names=['id', 'label'])
        
        # Format IDs to match file names
        self.records['id'] = self.records['id'].map(lambda i: f'{i:0>4}')
        
        # Create file paths
        self.paths = [os.path.join(self.folder_path, f'{filename}.npy') 
                     for filename in self.records['id'].tolist()]
        self.labels = self.records['label'].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.paths[index])
        
        if self.normalize:
            array = (array - array.min()) / (array.max() - array.min())
        
        if self.transform:
            array = self.transform(array)
        else:
            array = np.stack((array,)*3, axis=1)
            array = torch.FloatTensor(array)
            
        # Simplified label handling
        label = torch.FloatTensor([self.labels[index]])
            
        weight = torch.FloatTensor([1.0])
            
        return array, label, weight

class MRNet(nn.Module):
    """MRNet model architecture"""
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.alexnet(weights='DEFAULT')
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        # Changed from classifier to classifer to match saved state dict
        self.classifer = nn.Linear(256, 2)  

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        # Changed from classifier to classifer
        output = self.classifer(flattened_features)  
        return output

class GlobalACLClassifier:
    def __init__(self, models_dir, data_dir):
        """
        Initialize the global ACL classifier
        Args:
            models_dir: Directory containing trained models
            data_dir: Directory containing MRI data
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.planes = ['axial', 'coronal', 'sagittal']
        self.logreg = None
        
    def extract_predictions(self, task, plane, train=True):
        """
        Extract predictions from a specific plane model
        Args:
            task: Type of task ('acl', 'meniscus', 'abnormal')
            plane: Type of plane ('axial', 'coronal', 'sagittal')
            train: Whether to use training or validation data
        Returns:
            predictions and labels
        """
        assert task in ['acl', 'meniscus', 'abnormal']
        assert plane in self.planes
        
        model_files = list(self.models_dir.glob(f'*{task}*{plane}*.pth'))
        if not model_files:
            raise FileNotFoundError(f"No model found for task={task}, plane={plane}")
        model_path = model_files[0]
        
        mrnet = MRNet()
        mrnet.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        mrnet = mrnet.to(self.device)
        mrnet.eval()
        
        dataset = MRDataset(
            self.data_dir,
            task,
            plane,
            transform=None,
            train=train,
            normalize=False
        )
        
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )
        
        predictions = []
        labels = []
        
        with torch.no_grad():
            for image, label, _ in tqdm(loader, desc=f"Processing {plane}"):
                image = image.to(self.device)
                logit = mrnet(image)
                prediction = torch.sigmoid(logit)
                predictions.append(prediction[0][1].item())
                # Simplified label extraction
                labels.append(label.item())
                
        return predictions, labels
    
    def train(self, task='acl'):
        """
        Train the global classifier
        Args:
            task: Type of task to train for (default: 'acl')
        """
        print("Extracting training predictions...")
        results = {}
        
        # Get predictions from each plane
        for plane in self.planes:
            predictions, labels = self.extract_predictions(task, plane, train=True)
            results['labels'] = labels
            results[plane] = predictions
            
        # Prepare training data
        X = np.zeros((len(predictions), 3))
        for i, plane in enumerate(self.planes):
            X[:, i] = results[plane]
            
        y = np.array(results['labels'])
        
        # Train logistic regression
        print("Training logistic regression...")
        self.logreg = LogisticRegression(solver='lbfgs')
        self.logreg.fit(X, y)
        
        return self
    
    def evaluate(self, task='acl'):
        """
        Evaluate the global classifier
        Args:
            task: Type of task to evaluate (default: 'acl')
        Returns:
            AUC score
        """
        if self.logreg is None:
            raise ValueError("Model needs to be trained first")
            
        print("Extracting validation predictions...")
        results_val = {}
        
        # Get predictions from each plane
        for plane in self.planes:
            predictions, labels = self.extract_predictions(task, plane, train=False)
            results_val['labels'] = labels
            results_val[plane] = predictions
            
        # Prepare validation data
        X_val = np.zeros((len(predictions), 3))
        for i, plane in enumerate(self.planes):
            X_val[:, i] = results_val[plane]
            
        y_val = np.array(results_val['labels'])
        
        # Make predictions
        y_pred = self.logreg.predict_proba(X_val)[:, 1]
        auc_score = metrics.roc_auc_score(y_val, y_pred)
        
        print(f"Validation AUC: {auc_score:.3f}")
        return auc_score
    
    def predict(self, axial_scan, coronal_scan, sagittal_scan):
        """
        Make prediction on new scans
        Args:
            axial_scan: Axial plane scan
            coronal_scan: Coronal plane scan
            sagittal_scan: Sagittal plane scan
        Returns:
            Probability of ACL tear
        """
        if self.logreg is None:
            raise ValueError("Model needs to be trained first")
            
        # Get predictions from individual models
        scans = {
            'axial': axial_scan,
            'coronal': coronal_scan,
            'sagittal': sagittal_scan
        }
        
        X = np.zeros((1, 3))
        for i, (plane, scan) in enumerate(scans.items()):
            # Process scan and get prediction
            # This would need to be implemented based on how you want to handle
            # preprocessing of new scans
            pass
            
        # Make final prediction
        prob = self.logreg.predict_proba(X)[0, 1]
        return prob
    def save_global_classifier(self, save_path):
        """
        Save the trained logistic regression model
        Args:
            save_path: Path to save the model (e.g., 'models/global_classifier.pkl')
        """
        if self.logreg is None:
            raise ValueError("No trained model to save. Train the model first.")
        
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(self.logreg, f)
        print(f"Global classifier saved to {save_path}")

    def load_global_classifier(self, load_path):
        """
        Load a trained logistic regression model
        Args:
            load_path: Path to the saved model
        """
        import pickle
        with open(load_path, 'rb') as f:
            self.logreg = pickle.load(f)
        print(f"Global classifier loaded from {load_path}")

# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = GlobalACLClassifier(
        models_dir='./models/',
        data_dir='./data/'
    )
    
    # Train the global classifier
    classifier.train(task='acl')
    classifier.save_global_classifier('models/global_classifier.pkl')
    
    # Evaluate
    auc_score = classifier.evaluate(task='acl')