from torch.autograd import Variable
import os.path as osp
import os
from alfred.dl.torch.common import device
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from alfred.vis.image.seg import label_to_color_image
import cv2


class Trainer(object):

    def __init__(self, model, train_loader, val_loader, metric, class_weights, lr=5e-4, decay=0.1, epochs=1000,
                 out='checkpoints/model'):
        self.model = model
        self.epochs = epochs
        self.start_epoch = 0

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_weights = class_weights
        self.lr = lr
        self.decay = decay
        self.metric = metric
        self.best_iou = 0

        self.target_size = (640, 480)

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.save_path = osp.join(
            self.out, '{}_checkpoint.pth.tar'.format(self.out.split('/')[-1]))
        self.snap_img_path = os.path.join(self.out, 'snap_images')
        os.makedirs(self.snap_img_path, exist_ok=True)

        self.epoch = 0
        self._create_optimizer()
        self.load_checkpoint()

    def _create_optimizer(self):
        # optimizer
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=2e-4)
        self.lr_updater = lr_scheduler.StepLR(self.optimizer, 100, self.decay)
        if self.class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights, ignore_index=255).to(device)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)

    def train(self):
        self.model.train()
        try:
            for e in range(self.start_epoch, self.epochs):
                epoch_loss = 0
                if self.metric:
                    self.metric.reset()
                self.lr_updater.step()

                for i, (data, target) in enumerate(self.train_loader):
                    try:
                        # Wrap them in a Varaible
                        # print('label: ', target[0].numpy())
                        inputs, labels = Variable(
                            data.to(device)), Variable(target.to(device))
                        # print('inputs size: ', inputs.size())
                        # print('inputs : ', inputs)
                        outputs = self.model(inputs)
                        # print('output size: ', outputs.size())
                        # print('label size: ', labels.size())
                        loss = self.criterion(outputs, labels)

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        epoch_loss += loss.item()

                        if self.metric:
                            self.metric.add(outputs.data, labels.data)

                        if i % 50 == 0:
                            if self.metric:
                                print("Epoch: %d, iter %d, loss: %.4f, mean_iou: %.4f " % (e, i, loss.item(),
                                                                                           self.metric.value()[-1]))
                            else:
                                print("Epoch: %d, iter %d, loss: %.4f" %
                                      (e, i, loss.item()))
                    except Exception as e:
                        print('May got nan at loss. pass it. ')
                        print(e)
                        continue
                if self.metric:
                    print('Epoch {} finished, average loss is: {}, '
                          'meanIoU: {}\n'.format(e, epoch_loss / len(self.train_loader), self.metric.value()[-1]))
                else:
                    print('Epoch {} finished, average loss is: {}\n'.format(
                        e, epoch_loss / len(self.train_loader)))
                if self.metric:
                    if e % 5 == 0:
                        if self.metric.value()[-1] > self.best_iou:
                            self.best_iou = self.metric.value()[-1]
                            print('Got a best iou result of: {}, saving this model...'.format(
                                self.best_iou))
                            self.save_checkpoint(epoch=e, iter=i)
                if e % 2 == 0 and e != 0:
                    print('periodically saved model...')
                    self.save_checkpoint(epoch=e, iter=i)
                    # save the predicted segment mask
                    _, predictions = torch.max(outputs.data, 1)
                    prediction = predictions.cpu().numpy()[0]
                    if self.metric:
                        prediction = prediction - 1
                    print('prediction shape: ', prediction.shape)
                    mask_color = np.asarray(label_to_color_image(
                        prediction, 'cityscapes'), dtype=np.uint8)
                    mask_color = cv2.resize(mask_color, self.target_size)
                    cv2.imwrite(os.path.join(self.snap_img_path,
                                             '{}_mask.png'.format(e)), mask_color)
                    print(inputs.cpu().numpy()[0].shape)
                    frame = cv2.resize(np.transpose(
                        inputs.cpu().numpy()[0], (1, 2, 0)), self.target_size)
                    frame = np.array(frame*255, dtype=np.uint8)
                    print('frame vs mask: {} {}'.format(
                        mask_color.shape, frame.shape))
                    res = cv2.addWeighted(frame, 0.5, mask_color, 0.7, 1)
                    cv2.imwrite(os.path.join(self.snap_img_path,
                                             '{}_combined.png'.format(e)), res)
                    cv2.imwrite(os.path.join(self.snap_img_path,
                                             '{}_image.png'.format(e)), frame)

        except KeyboardInterrupt:
            print('Try saving model, pls hold...')
            self.save_checkpoint(epoch=e, iter=i)
            print('Model has been saved into: {}'.format(self.save_path))

    def save_checkpoint(self, epoch, iter, is_best=True):
        torch.save({
            'epoch': epoch,
            'iteration': iter,
            'arch': self.model.__class__.__name__,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': self.model.state_dict(),
            'miou': self.best_iou,
        }, self.save_path)

    def load_checkpoint(self):
        if not self.out:
            os.makedirs(self.out, exist_ok=True)
        else:
            if os.path.exists(self.save_path):
                print('Loading checkpoint {}'.format(self.save_path))
                checkpoint = torch.load(self.save_path)
                self.start_epoch = checkpoint['epoch']
                # self.best_top1 = checkpoint['best_top1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.best_iou = checkpoint['miou']
                print('checkpoint loaded successful from {} at epoch {}, best mean iou: {}'.format(
                    self.save_path, self.start_epoch, self.best_iou
                ))
            else:
                print('No checkpoint exists from {}, skip load checkpoint...'.format(
                    self.save_path))
