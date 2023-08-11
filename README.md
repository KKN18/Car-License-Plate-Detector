# Car-License-Plate-Detector
AI model detecting and recognizing Car License Plate

## Environment
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XfnsxGwYLLVFc-uhnJNBM5um8VTBuaIK#scrollTo=B031ZvmvMiqu)

## Dataset
<img width="500" src="https://user-images.githubusercontent.com/63842546/213862572-89924584-77c7-448d-b8f8-c8525c66980f.JPG"/>
[Kaggle Dataest](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)

## Model
### Predict each 4 corner of License Plate


```python
model = models.resnet101(pretrained=True)
model = nn.Sequential(
    model,
    nn.Linear(1000, 500),
    nn.ReLU(),
    nn.Linear(500, 250),
    nn.ReLU(),
    nn.Linear(250, 4),   # 4 labels : (xmin, xmax, ymin, ymax)
    nn.Sigmoid(),
)
```


### Resnet-101 

<img width="500" src="https://user-images.githubusercontent.com/63842546/213862967-dc11e2cc-8aad-4d3d-98e3-9bea79a7cbb3.png"/>

## Detection
### Original Image
<img width="500" src="https://user-images.githubusercontent.com/63842546/213863332-822def14-f26c-42cc-baac-61c8c38c8f93.png"/>

### Cropped Image
<img width="100" src="https://user-images.githubusercontent.com/63842546/213863313-41a8e8f2-9b0c-4ff4-a765-1b17d6ed5c42.png"/>

## Recognition
### Pytesseract OCR
<img width="400" src="https://user-images.githubusercontent.com/63842546/213863488-6c348fe3-f8c9-4a08-8e12-cd92dcde0679.png"/>

```python
import pytesseract as pt
text = pt.image_to_string(plateImage)
print(text)
```
### Target Image and Recognition
<img width="100" src="https://user-images.githubusercontent.com/63842546/213863313-41a8e8f2-9b0c-4ff4-a765-1b17d6ed5c42.png"/>
 Tcs26 JHD P

