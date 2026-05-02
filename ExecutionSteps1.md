import torch
import torchvision
import openvino as ov
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 1. Define the Wrapper to fix the "Dictionary Output" error
class FasterRCNNWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image_tensor):
        # Faster R-CNN expects a list of tensors [C, H, W]
        # Our input is [1, C, H, W], so we take the first element
        res = self.model(image_tensor) 
        # Extract tensors from the dictionary so OpenVINO is happy
        # We return them as a tuple: (boxes, labels, scores)
        return res[0]["boxes"], res[0]["labels"], res[0]["scores"]

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    print("--- Starting Fixed OpenVINO Conversion ---")
    
    # Load original model
    base_model = get_model(num_classes=11)
    base_model.load_state_dict(torch.load('fasterrcnn_curriculum_final.pth', map_location='cpu'))
    base_model.eval()

    # Wrap it
    wrapped_model = FasterRCNNWrapper(base_model)
    wrapped_model.eval()

    # Define dummy input [Batch, Channels, Height, Width]
    example_input = torch.randn(1, 3, 240, 426)

    try:
        print("Converting wrapped model...")
        ov_model = ov.convert_model(wrapped_model, example_input=example_input)
        
        # Save it
        ov.save_model(ov_model, 'faster_rcnn_openvino.xml')
        print("✅ SUCCESS! Created 'faster_rcnn_openvino.xml'")
        print("Now use the demo script to run it.")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")

if __name__ == "__main__":
    main()