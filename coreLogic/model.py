from pathlib import Path
import subprocess

BASE_DIR = Path(__file__).resolve().parent

def LaneNet_TensorFlow(folder_image):
    folder_model = BASE_DIR / "Model/lanenet-lane-detection"
    weight = "tusimple_val_miou=0.6590.ckpt-872"
    python_env = fr"C:\Users\Parim\anaconda3\envs\LaneNet-TensorFlow\python.exe"

    result = subprocess.run([
        python_env,
        "-m",
        "tools.test_lanenet",
        "--weights_path",
        f"weights/tusimple_lanenet/{weight}",
        "--image_path",
        f"{BASE_DIR}/streetview_output/{folder_image}",
        "--road_name",
        folder_image
    ],
    cwd=folder_model,
    text=True,
    capture_output=True
    )
    print(result.stdout)
    print(result.stderr)
    

def LaneNet_PyTorch(folder_image):
    folder_model = BASE_DIR / "Model/lanenet-lane-detection-pytorch"
    python_env = fr"C:\Users\Parim\anaconda3\envs\LaneNet-TensorFlow\python.exe"

    result = subprocess.run([
        python_env,
        "test.py",
        "--img",
        f"{BASE_DIR}/streetview_output/{folder_image}",
        "--road_name", folder_image
    ],
    cwd=folder_model,
    text=True,
    capture_output=True
    )
    print(result.stdout)
    print(result.stderr)

def DeepLabV3(folder_image):
    folder_model = BASE_DIR / "Model/deeplabv3"
    weight = "best_deeplab_lane.pth"
    python_env = fr"C:\Users\Parim\anaconda3\envs\deeplabv3\python.exe"

    result = subprocess.run([
            python_env,
            "test.py",
            "--weights_path", weight,
            "--image_path", folder_image
        ],
        cwd=folder_model,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    print(result.stderr)

def UNet(folder_image):
    folder_model = BASE_DIR / "Model/unet"
    weight = "segment_deeplab.pth"
    python_env = fr"C:\Users\Parim\anaconda3\envs\deeplabv3\python.exe"

    result = subprocess.run([
            python_env,
            "test_unet.py",
            "--weights_path", weight,
            "--image_path", folder_image
        ],
        cwd=folder_model,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    print(result.stderr)
    
    

