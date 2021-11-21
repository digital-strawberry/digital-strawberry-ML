import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn


class RulerService:
    def __init__(
            self,
            growth_measure_path: str = "growth_levels"):
        self.RESCALE_SIZE = 224
        self.growth_measure_path = growth_measure_path
        self._levelmeasure = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.encoder = [
            'начало роста',
            'появление первых 2 - 4 листьев',
            'появление цветоноса',
            'полный выход цветоносов',
            'цветение',
            'опадение лепестков',
            'формирование ягод',
            'развитие плода',
            'дифференциация почек и усов'
        ]

    def prepare_photo(self, photo_path: str) -> np.array:
        image = Image.open(photo_path)
        image.load()
        image = image.resize((self.RESCALE_SIZE, self.RESCALE_SIZE))
        image = np.array(image)
        image = np.array(image / 255, dtype='float32')
        image = self.transform(image)
        return image.unsqueeze(0)

    def predict_level(self, photo_path: str) -> str:
        return self.encoder[round(self.levelmeasure(self.prepare_photo(photo_path)).item())]

    # load levelmeasure model lazily for graphic memory saving
    @property
    def levelmeasure(self):
        if self._levelmeasure is None:
            self._set_levelmeasure()
        return self._levelmeasure

    def _set_levelmeasure(self):
        self._levelmeasure = models.densenet121(pretrained=True)
        num_ftrs = self._levelmeasure.classifier.in_features
        self._levelmeasure.classifier = nn.Linear(num_ftrs, 1)
        self._levelmeasure.load_state_dict(torch.load(self.growth_measure_path, map_location=torch.device('cpu')))
        self._levelmeasure.eval()


'''
USAGE EXAMPLE

Ruler = RulerService()
print(Ruler.predict_level('3.jpeg'))
'''
