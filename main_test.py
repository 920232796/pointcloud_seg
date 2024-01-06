import os
import cv2
import numpy as np
import argparse
import time
import torch
from KittiCalibration import KittiCalibration
from visualizer import Visualizer
from pointpainting import PointPainter
from bev_utils import boundary
from matplotlib.colors import ListedColormap
from PIL import Image
from torchvision import transforms
dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

print(f"device is {dev}")

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main(args):

    painter = PointPainter()

    # visualizer = Visualizer(args.mode)

    image = cv2.imread(args.image_path)
    pointcloud = np.fromfile(args.pointcloud_path, dtype=np.float32).reshape((-1, 4))

    calib = KittiCalibration(args.calib_path)

    from seg_model.segformer import Segmenrator
    segmentor = Segmenrator("pytorch",
                            1,
                            1,
                            device=device,
                            num_gpus=0
                            )
    model = segmentor.model

    original_size = image.shape[:2]
    print(original_size)
    image_pre = Image.fromarray(image)
    image_pre =image_pre.resize((512, 512))

    img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    image_pre = img_transform(image_pre)

    image_pre = image_pre[None,]

    print(image_pre.shape)
    with torch.no_grad():
        pred = model(image_pre)
    pred = torch.nn.functional.interpolate(pred, size=original_size,  mode="bilinear")
    pred = pred.argmax(dim=1)[0]

    pred = np.int64(pred.cpu().numpy())

    semantic = pred

    import matplotlib.pyplot as plt 

    colors = np.random.rand(150, 3)
    cmap = ListedColormap(colors)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(semantic, cmap=cmap)
    plt.savefig(args.save_semantic_path)
    # plt.show()

    semantic = np.int64(semantic)
    painted_pointcloud = painter.paint(pointcloud, semantic, calib)

    np.save(args.save_pointcloud_path, painted_pointcloud)
    print(np.unique(painted_pointcloud[:, 3]))
    print(painted_pointcloud.shape)
    print("~~~~~~~~~~~~~~~")
    #import pandas as pd
    #df = pd.DataFrame(painted_pointcloud[:,3])
    #print(df.value_counts())
 
    # color_image = visualizer.get_colored_image(image, semantic)
    # cv2.imshow("scene", color_image)
    # visualizer.visuallize_pointcloud(painted_pointcloud, blocking=True)
  
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default=r'KIN.png')
    parser.add_argument('--pointcloud_path', type=str, default=r'KIN.bin')
    parser.add_argument('--calib_path', type=str, default=r'calib_kitti.txt')
    parser.add_argument('--weights_path', type=str, default='BiSeNetv2/checkpoints/BiseNetv2_150.pth',)
    parser.add_argument('--save_semantic_path', type=str, default=r'semantic.png',)
    parser.add_argument('--save_pointcloud_path', type=str, default='pointcloud.bin',)
    parser.add_argument('--mode', type=str, default='2d', choices=['2d', '3d'],
    help='visualization mode .. img is semantic image .. 2d is semantic + bev .. 3d is colored pointcloud')

    args = parser.parse_args()
    main(args)



