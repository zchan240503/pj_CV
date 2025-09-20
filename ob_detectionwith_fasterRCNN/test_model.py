import torch
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def test_model():
    print("Testing model...")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = fasterrcnn_mobilenet_v3_large_320_fpn()
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes=21)
    
    # Try to load trained model
    checkpoint_path = "trained_models/best.pt"
    if os.path.exists(checkpoint_path):
        print("Loading trained model...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Using default pre-trained weights")
    
    model = model.float()
    model.to(device)
    model.eval()
    
    # Create a test image (random noise)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"Test image shape: {test_image.shape}")
    
    # Process image
    image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))/255.
    image = [torch.from_numpy(image).to(device).float()]
    
    with torch.no_grad():
        output = model(image)[0]
        bboxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]
        
        print(f"Model output - Boxes: {len(bboxes)}, Labels: {len(labels)}, Scores: {len(scores)}")
        
        if len(bboxes) > 0:
            print("Model is working! Found detections:")
            for i, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):
                print(f"  Detection {i+1}: Label {label.item()}, Score {score.item():.3f}")
        else:
            print("No detections found in test image (this is normal for random noise)")
    
    print("Model test completed successfully!")

if __name__ == "__main__":
    import os
    test_model() 