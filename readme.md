# DeepLabV3 with MobileNetV2



This is the **pytorch** implementation of DeepLabV3 segmentation with MobileNetV2 support backbone. It achieve STOA speed and meanIOU on semantic segmentation. Benefit from MobileNetV2 depth-wise convolution and DeepLabV3 the most advanced ASPP module, the segmentation result is remarkable.  Here is some screen shot of result:

![](https://s2.ax1x.com/2019/01/08/Fq8vm4.png)



This is only about 23 epoch result, further result maybe update later. For now, DeepLabV3 with MobileNetV2 has those features and you can not reject it:



- Fast: almost 25 fps on GTX1080, it's almost **80%** faster than original DeeplabV3;
- Accurate: compare to ENet or SegNet or UNet or RetinaSeg, it achieve almost 78 meanIOU on test dataset;
- Without post process with good result, as you can see, the result can almost use without CRF post process.



## Install

To run:



```
sudo pip3 install alfred-py
python3 demo.py
```





## Further training

For training, you can obtain full version from [StrangeAI](http://strangeai.pro)



## Contact

Any question could be asked via Setu(a secret chat app): http://loliloli.pro