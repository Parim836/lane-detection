import time
import os
import sys

import torch
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from model.utils.cli_helper_test import parse_args
import numpy as np
from PIL import Image
import pandas as pd
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_test_data(img_path, transform):
    img = Image.open(img_path)
    img = transform(img)
    return img

def test():
    args = parse_args()
    road_name = args.road_name
    img_path = args.img
    resize_height = args.height
    resize_width = args.width
    
    if os.path.exists('output') == False:
        os.mkdir('output')
    output_root = 'output'

    binary_dir = os.path.join(output_root, road_name)
    # instance_dir = os.path.join(output_root, 'instance')
    # input_dir = os.path.join(output_root, 'input')

    os.makedirs(binary_dir, exist_ok=True)
    # os.makedirs(instance_dir, exist_ok=True)
    # os.makedirs(input_dir, exist_ok=True)
    

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = args.model
    model = LaneNet(arch=args.model_type)
    # state_dict = torch.load(model_path)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    # dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
    # dummy_input = torch.unsqueeze(dummy_input, dim=0)
    # outputs = model(dummy_input)

    # input = Image.open(img_path)
    # input = input.resize((resize_width, resize_height))
    # input = np.array(input)

    # instance_pred = torch.squeeze(outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
    # binary_pred = torch.squeeze(outputs['binary_seg_pred']).to('cpu').numpy() * 255

    # cv2.imwrite(os.path.join('test_output', 'input.jpg'), input)
    # cv2.imwrite(os.path.join('test_output', 'instance_output.jpg'), instance_pred.transpose((1, 2, 0)))
    # cv2.imwrite(os.path.join('test_output', 'binary_output.jpg'), binary_pred)

    if os.path.isdir(img_path):
        image_list = [
            os.path.join(img_path, f)
            for f in os.listdir(img_path)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
    else:
        image_list = [img_path]

    for img_file in image_list:
        print("Processing:", img_file)

        dummy_input = load_test_data(img_file, data_transform).to(DEVICE)
        dummy_input = torch.unsqueeze(dummy_input, dim=0)

        # outputs = model(dummy_input) #รันแบบมี instance
        with torch.inference_mode():  #รัันแค่ binary
            outputs = model(dummy_input, binary_only=True)  

        input_img = Image.open(img_file)
        input_img = input_img.resize((resize_width, resize_height))
        input_img = np.array(input_img)

        # instance_pred = torch.squeeze( #รันแบบมี instance
        #     outputs['instance_seg_logits'].detach().cpu()
        # ).numpy() * 255

        binary_pred = torch.squeeze(
            outputs['binary_seg_pred']
        ).cpu().numpy() * 255

        filename = os.path.splitext(os.path.basename(img_file))[0]

        # cv2.imwrite(os.path.join(input_dir, f'{filename}.jpg'), input_img)
        # cv2.imwrite(os.path.join(instance_dir, f'{filename}.jpg'), instance_pred.transpose((1, 2, 0))) #รันแบบมี instance
        cv2.imwrite(os.path.join(binary_dir, f'{filename}.jpg'), binary_pred)


if __name__ == "__main__":
    test()