import torch
import numpy as np
import matplotlib.pyplot as plt
from model import MRNet

class CAMGenerator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        self.weights = model.classifer.weight.data

    def generate_cam(self, image):
        """
        Generate class activation map
        Args:
            image: Single slice of MRI (2D numpy array)
        """
        # Ensure image is normalized
        if image.dtype != np.float32:
            image = (image - image.min()) / (image.max() - image.min())
        
        # Create correct input tensor format (batch_size, channels, height, width)
        image_tensor = torch.FloatTensor(image).unsqueeze(0)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.repeat(1, 3, 1, 1)
        
        # Move tensor to same device as model
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            # Get features
            features = self.model.pretrained_model.features(image_tensor)
            
            # Get the activations and weights
            activations = self.model.pooling_layer(features)
            activations = activations.view(activations.size(0), -1)
            
            # Create CAM
            cam = features.mean(dim=1)
            cam = cam.squeeze(0)
            
            # Move CAM back to CPU for processing
            cam = cam.cpu()
            
            # Normalize CAM
            cam = (cam - cam.min()) / (cam.max() - cam.min())
            
            # Upsample to original size
            cam = torch.nn.functional.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=(image.shape[0], image.shape[1]),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            return cam.numpy()

def visualize_cam(image, cam, save_path=None):
    plt.figure(figsize=(10, 5))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot CAM overlay
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

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = MRNet()
    model.load_state_dict(
        torch.load('models/model_acl_axial_acl_coronal_val_auc_0.8042_train_auc_0.7960_epoch_13.pth',
                  map_location=device,
                  weights_only=True)
    )
    model = model.to(device)
    
    # Load and preprocess image
    mri_volume = np.load('./data/train/coronal/0001.npy')
    
    # Generate CAM for middle slice
    middle_slice_idx = len(mri_volume) // 2
    middle_slice = mri_volume[middle_slice_idx]
    
    # Create CAM generator and generate CAM
    cam_gen = CAMGenerator(model, device)
    cam = cam_gen.generate_cam(middle_slice)
    
    # Visualize
    visualize_cam(middle_slice, cam, 'cam_visualization.png')
    
    print("CAM visualization saved as cam_visualization.png")

if __name__ == "__main__":
    main()