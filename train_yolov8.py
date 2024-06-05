import torch
from ultralytics import YOLO

def main():
    # Check if CUDA is available and print the device being used
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = YOLO('yolov8n.yaml')
    model.train(data='dataset/data.yaml', epochs=125, imgsz=640, batch=16, device=device)

if __name__ == '__main__':
    main()