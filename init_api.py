from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import pydicom
import numpy as np
import torch
from model import MRNet
import io
from PIL import Image
import os
import matplotlib.pyplot as plt
from typing import List
import uvicorn

app = FastAPI()

class CAMGenerator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        self.weights = model.classifer.weight.data

    def generate_cam(self, image):
        # Ensure image is normalized
        if image.dtype != np.float32:
            image = (image - image.min()) / (image.max() - image.min())
        
        # Create correct input tensor format
        image_tensor = torch.FloatTensor(image).unsqueeze(0)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.repeat(1, 3, 1, 1)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            features = self.model.pretrained_model.features(image_tensor)
            activations = self.model.pooling_layer(features)
            activations = activations.view(activations.size(0), -1)
            
            cam = features.mean(dim=1)
            cam = cam.squeeze(0).cpu()
            
            cam = (cam - cam.min()) / (cam.max() - cam.min())
            
            cam = torch.nn.functional.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=(image.shape[0], image.shape[1]),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            return cam.numpy()

def visualize_cam(image, cam, save_path=None):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title('Class Activation Map')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = {
        'axial': MRNet(),
        'coronal': MRNet(),
        'sagittal': MRNet()
    }
    
    model_files = {
        'axial': 'model_acl_axial_acl_axial_val_auc_0.8342_train_auc_0.7937_epoch_11.pth',
        'coronal': 'model_acl_axial_acl_coronal_val_auc_0.8042_train_auc_0.7960_epoch_13.pth',
        'sagittal': 'model_acl_sagittal_acl_sagittal_val_auc_0.8426_train_auc_0.7770_epoch_13.pth'
    }
    
    for plane, model in models.items():
        model.load_state_dict(
            torch.load(f'models/{model_files[plane]}',
                      map_location=device,
                      weights_only=True)
        )
        model = model.to(device)
        model.eval()
        
    return models, device

def dicom_to_numpy(dicom_file) -> np.ndarray:
    """Convert DICOM file to numpy array"""
    ds = pydicom.dcmread(dicom_file)
    return ds.pixel_array

def prepare_input(array):
    """Prepare input for model"""
    array = (array - array.min()) / (array.max() - array.min())
    array = np.stack((array,)*3, axis=1)
    array = torch.FloatTensor(array)
    array = array.unsqueeze(0)
    return array

# Initialize models at startup
models, device = load_models()

@app.post("/predict_acl")
async def predict_acl(
    axial_file: UploadFile = File(...),
    coronal_file: UploadFile = File(...),
    sagittal_file: UploadFile = File(...)):
    
    try:
        # Read DICOM files
        axial_array = dicom_to_numpy(io.BytesIO(await axial_file.read()))
        coronal_array = dicom_to_numpy(io.BytesIO(await coronal_file.read()))
        sagittal_array = dicom_to_numpy(io.BytesIO(await sagittal_file.read()))
        
        # Process each plane
        predictions = {}
        cam_paths = []
        
        for plane, array in [
            ('axial', axial_array),
            ('coronal', coronal_array),
            ('sagittal', sagittal_array)
        ]:
            # Get prediction
            input_tensor = prepare_input(array).to(device)
            with torch.no_grad():
                output = models[plane](input_tensor)
                prob = torch.sigmoid(output)[0][1].item()
                predictions[plane] = prob
            
            # Generate CAM
            middle_slice_idx = len(array) // 2
            middle_slice = array[middle_slice_idx]
            
            cam_gen = CAMGenerator(models[plane], device)
            cam = cam_gen.generate_cam(middle_slice)
            
            # Save CAM visualization
            output_path = f"temp_cam_{plane}.png"
            visualize_cam(middle_slice, cam, output_path)
            cam_paths.append(output_path)
        
        # Get final prediction using global classifier
        import pickle
        with open('models/global_classifier.pkl', 'rb') as f:
            global_classifier = pickle.load(f)
        
        plane_probs = [predictions[plane] for plane in ['axial', 'coronal', 'sagittal']]
        final_prob = global_classifier.predict_proba(np.array(plane_probs).reshape(1, -1))[0][1]
        
        # Prepare response
        response = {
            "final_probability": float(final_prob),
            "plane_probabilities": {
                plane: float(prob) for plane, prob in predictions.items()
            },
            "cam_visualizations": {
                plane: FileResponse(f"temp_cam_{plane}.png")
                for plane in ['axial', 'coronal', 'sagittal']
            }
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary files
        for path in cam_paths:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)