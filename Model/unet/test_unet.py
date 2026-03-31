import segmentation_models_pytorch as smp
from pathlib import Path
import argparse
import torch
import cv2
import os

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)

    return parser.parse_args()

def main():
    args = parse_args()

    device = "cpu"

    test_dir = PROJECT_DIR / f"streetview_output/{args.image_path}"
    save_dir = BASE_DIR / f"output/{args.image_path}"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cpu")
    
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)
    
    # --- โหลด model ---
    model.load_state_dict(torch.load("segment_deeplab.pth", map_location=device))
    model.to(device)
    model.eval()

    for img_name in os.listdir(test_dir):
        if not img_name.lower().endswith((".jpg",".png",".jpeg")):
            continue
        img_path = os.path.join(test_dir, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512,256))

        x = torch.tensor(img.transpose(2,0,1)/255.).float().unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x)
            mask = (pred.sigmoid()[0,0] > 0.5).cpu().numpy()

        mask_img = (mask*255).astype("uint8")

        base, ext = os.path.splitext(img_name)
        save_path = os.path.join(save_dir, base + ext)

        cv2.imwrite(save_path, mask_img)


if __name__ == "__main__":
    main()