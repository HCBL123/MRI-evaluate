import numpy as np
import pydicom
import torch
from model import MRNet
import os

def dicom_to_numpy(dicom_file):
    """Convert DICOM file to numpy array"""
    ds = pydicom.dcmread(dicom_file)
    array = ds.pixel_array
    # Normalize array
    array = (array - array.min()) / (array.max() - array.min())
    return array

def prepare_input(array):
    """Prepare input for model"""
    # Stack same image 3 times for RGB channels
    array = np.stack((array,)*3, axis=1)
    # Convert to tensor
    array = torch.FloatTensor(array)
    # Add batch dimension
    array = array.unsqueeze(0)
    return array

def predict_case(axial_path, coronal_path, sagittal_path, models_dir='./models/'):
    """
    Make prediction using all three planes
    Returns probability of ACL tear
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define specific model files for each plane
    model_files = {
        'axial': 'model_acl_axial_acl_axial_val_auc_0.8342_train_auc_0.7937_epoch_11.pth',
        'coronal': 'model_acl_axial_acl_coronal_val_auc_0.8042_train_auc_0.7960_epoch_13.pth',
        'sagittal': 'model_acl_sagittal_acl_sagittal_val_auc_0.8426_train_auc_0.7770_epoch_13.pth'
    }
    
    # Load and process DICOM files
    planes = {
        'axial': dicom_to_numpy(axial_path),
        'coronal': dicom_to_numpy(coronal_path),
        'sagittal': dicom_to_numpy(sagittal_path)
    }
    
    predictions = []
    
    # Get predictions from each plane
    for plane, array in planes.items():
        model_path = os.path.join(models_dir, model_files[plane])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        # Load model
        model = MRNet()
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model = model.to(device)
        model.eval()
        
        # Prepare input
        input_tensor = prepare_input(array)
        input_tensor = input_tensor.to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output)[0][1].item()
            predictions.append(prob)
            print(f"{plane} plane prediction: {prob:.3f}")
    
    # Load global classifier
    import pickle
    with open(os.path.join(models_dir, 'global_classifier.pkl'), 'rb') as f:
        global_classifier = pickle.load(f)
    
    # Get final prediction
    final_prob = global_classifier.predict_proba(np.array(predictions).reshape(1, -1))[0][1]
    return final_prob

# Example usage
def main():
    axial_path = 'output_dicoms/axial_multiframe.dcm'
    coronal_path = 'output_dicoms/coronal_multiframe.dcm'
    sagittal_path = 'output_dicoms/sagittal_multiframe.dcm'
    
    final_probability = predict_case(axial_path, coronal_path, sagittal_path)
    print(f"\nFinal probability of ACL tear: {final_probability:.3f}")

if __name__ == "__main__":
    main()