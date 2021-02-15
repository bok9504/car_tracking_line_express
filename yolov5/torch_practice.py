import torch
import cv2
from IPython.display import Image, clear_output
from PIL import Image
from utils.autoanchor import *;

anchor_size = kmean_anchors(path='./yolov5/data/data.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True)


# clear_output()

# # 토치 사용 확인
# print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

# # Start tensorboard
# # %load_ext tensorboard
# # %tensorboard --logdir runs

# # Model : 파이토치허브에서 바로 불러오는 방법
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()  # for PIL/cv2/np inputs and NMS

# # Images
# # for f in ['ShinozakiAi.jpg']:  # download 2 images
# #     print(f'Downloading {f}...')
# #     torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/' + f, f)
# img1 = Image.open('ShinozakiAi.jpg')  # PIL image
# # img2 = cv2.imread('bus.jpg')[:, :, ::-1]  # OpenCV image (BGR to RGB)
# imgs = [img1] # [img1, img2]  # batched list of images

# # Inference
# results = model(imgs, size=640)  # includes NMS : NMS (non-maximum-suppression)란 현재 픽셀을 기준으로 주변의 픽셀과 비교했을 때 최대값인 경우 그대로 놔두고, 아닐 경우(비 최대) 억제(제거)하는 것이다. 
# # Results
# results.print()  # print results to screen
# results.show()  # display results
# results.save()  # save as results1.jpg, results2.jpg... etc.