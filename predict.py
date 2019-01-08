from network import MobileNetv2_DeepLabv3Model
from config import Params
import torch
from alfred.dl.torch.common import device
import time
import cv2
from torch.autograd import Variable

from PIL import Image
import torchvision.transforms as transforms
import numpy as np

from alfred.vis.image.seg import label_to_color_image

image_size = 512


def predict():

    params = Params()

    model_path = 'checkpoints/Checkpoint_epoch_150.pth.tar'
    model = MobileNetv2_DeepLabv3Model(params).to(device)
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    new_dict = {}
    for k in state_dict.keys():
        # print('K: {}, '.format(k))
        new_dict['model.'+k] = state_dict[k]
    model.load_state_dict(new_dict)
    model.eval()
    print('Model loaded.')

    img_fs = [
        'images/berlin_000004_000019_leftImg8bit.png',
        'images/berlin_000002_000019_leftImg8bit.png', 
    ]
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    for img_f in img_fs:
        img = cv2.imread(img_f)
        inp = Variable(transform(Image.fromarray(img)).to(device).unsqueeze(0))
        print(inp.size())
        out = model(inp)
        print(out.size())

        _, predictions = torch.max(out.data, 1)
        prediction = predictions.cpu().numpy()[0]
        print(prediction)
        mask_color = np.asarray(label_to_color_image(prediction, 'cityscapes'), dtype=np.uint8)
        frame = cv2.resize(img, (1024, 512))
        print('msk: {}, frame: {}'.format(mask_color.shape, frame.shape))
        res = cv2.addWeighted(frame, 0.5, mask_color, 0.7, 1)

        cv2.imshow('res', res)
        while True:
            cv2.waitKey(27)


if __name__ == "__main__":
    predict()

