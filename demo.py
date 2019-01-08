"""
Demo file shows how to inference semantic segmentation
with cityscapes or other trained model in ENet method

"""
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from models.deeplabv3_mb2 import DeepLabV3MobileNetV2
from alfred.dl.torch.common import device
import cv2
from PIL import Image
import numpy as np
from alfred.vis.image.get_dataset_colormap import label_to_color_image
from alfred.dl.inference.image_inference import ImageInferEngine


class DeepLabV3MobileNetV2Demo(ImageInferEngine):

    def __init__(self, f, model_path, enable_crf=False):
        super(DeepLabV3MobileNetV2Demo, self).__init__(f=f)

        self.target_size = (512, 1024)
        self.model_path = model_path
        self.num_classes = 20

        self.image_transform = transforms.Compose(
            [transforms.Resize(self.target_size),
             transforms.ToTensor()])

        self.enable_crf = enable_crf

        self._init_model()

    def _init_model(self):
        self.model = DeepLabV3MobileNetV2(self.num_classes).to(device)
        self.model.eval()
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        print('Model loaded!')
    
    def read_image_file(self, img_f):
        img = cv2.imread(img_f)
        return img

    def solve_a_image(self, img):
        images = Variable(self.image_transform(Image.fromarray(img)).to(device).unsqueeze(0))
        predictions = self.model(images)

        if not self.enable_crf:
            _, predictions = torch.max(predictions.data, 1)
            prediction = predictions.cpu().numpy()[0]
            return prediction
        else:
            # unary = np.transpose(predictions.cpu().detach().numpy(), [1, 2, 0])
            print('post process not implemented yet.')

    def vis_result(self, img, net_out):
        mask_color = np.asarray(label_to_color_image(net_out, 'cityscapes'), dtype=np.uint8)
        frame = cv2.resize(img, (self.target_size[1], self.target_size[0]))
        # mask_color = cv2.resize(mask_color, (frame.shape[1], frame.shape[0]))
        res = cv2.addWeighted(frame, 0.5, mask_color, 0.7, 1)
        cv2.imwrite('res.png', res)
        return res


if __name__ == '__main__':
    # v_f = '/media/jintain/sg/permanent/datasets/Cityscapes/videos/combined_stuttgart_01.mp4'
    v_f = 'images/berlin_000004_000019_leftImg8bit.png'
    enet_seg = DeepLabV3MobileNetV2Demo(f=v_f, model_path='checkpoints/deeplabv3_mobilenetv2_cityscapes_with_trans_512x1024_checkpoint.pth.tar', 
    enable_crf=False)
    enet_seg.run()

