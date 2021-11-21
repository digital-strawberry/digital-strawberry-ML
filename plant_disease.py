import torch
import numpy as np
import collections

from PIL import Image
import cv2

from torchvision import models, transforms
import torchvision.transforms.functional as F
import torch.nn as nn

from typing import Dict, List, Set


class PlantDoctor():
    def __init__(self,
                 binary_classifier_path: str = 'densenet_100',
                 diseases_classifier_path: str = 'densenet_diseases_clf_224_96'
                 ):
        self.RESCALE_SIZE = 224
        self.set_healthcheck(binary_classifier_path)
        self.set_diseases_classifier(diseases_classifier_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.label_encoder = ['blight', 'measles', 'mold', 'powdery_meldew', 'rot', 'scorch', 'spider', 'spot', 'virus']
        self.alchemy = {
            'mold': {'фунгициды'},
            'powdery-meldew': {'раствор кальцинированной соды', 'молочная сыворотка', 'раствор йода'},
            'rot': {'фунгициды', 'бордоская смесь'},
            'spider': {'пестициды'},
            'spot': {'медные удобрения'}
        }
        self.cure_cookbook = {
            'blight': {'temp-lo', 'hum-lo'},
            'measles': {'hum-lo', 'light-hi', 'azot-lo', 'insects', 'weed'},
            'mold': {'hum-lo', 'temp-hi', 'air-hi', 'light-hi', 'chem'},
            'powdery_meldew': {'temp-lo', 'hum-lo', 'chem'},
            'rot': {'temp-lo', 'hum-lo', 'chem'},
            'scorch': {'weed', 'air-hi', 'hum-lo'},
            'spider': {'azot-lo', 'water-hi', 'temp-lo', 'chem'},
            'spot': {'chem', 'hum-lo'},
            'virus': {'insects'}
        }

        self.measure_dict = {
            'temp': 'теспературу',
            'light': 'освещённость',
            'air': 'циркуляцию воздуха',
            'hum': 'влажность',
            'azot': 'количество азотных удобрений',
            'water': 'частоту и объём полива'
        }

    def prepare_photo(self, photo_path: str) -> np.array:
        image = Image.open(photo_path)
        image.load()
        image = image.resize((self.RESCALE_SIZE, self.RESCALE_SIZE))
        image = np.array(image)
        image = np.array(image / 255, dtype='float32')
        image = self.transform(image)
        return image.unsqueeze(0)

    def prepare_square(self, square: np.array) -> np.array:
        image = np.array(square / 255, dtype='float32')
        image = self.transform(image)
        return image.unsqueeze(0)

    def diagnose_fragment(self, photo_path: str) -> np.array:
        health_distribution = np.zeros(9)
        photo = self.prepare_photo(photo_path)

        if not self.is_healthy(photo):
            health_distribution = self.get_health_distribution(photo)

        return health_distribution

    def scale(self, img, new_h):
        w, h = img.size
        mx = max(w, h)
        wp = int(mx - w)
        hp = int(mx - h)
        padding = (0, 0, wp, hp)
        img = F.pad(img, padding, 0, 'constant')
        img = F.resize(img, (new_h, new_h))
        return img

    def leaf_squares(self, path):
        h = 224
        pil_img = Image.open(path)
        img = self.scale(pil_img, 1792)
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        total = h ** 2

        for y in range(0, 1792, h):
            for x in range(0, 1792, h):
                cut = img[y:y + h, x:x + h]
                hsv = cv2.cvtColor(cut, cv2.COLOR_BGR2HSV)
                green = cv2.inRange(hsv[:, :], (24, 50, 15), (90, 230, 230))
                stat = sum(collections.Counter(x)[255] for x in green)
                percentage = stat / total
                if percentage > 0.4:
                    yield cut

    def diagnose(self, filepath: str) -> Dict:
        total_distribution = np.zeros(9)
        square_counter = 0
        for square in self.leaf_squares(filepath):
            picture = self.prepare_square(square)
            total_distribution += self.get_health_distribution(picture)
            square_counter += 1
        total_distribution /= square_counter if square_counter else 0.9

        health_rate = 1 - max(total_distribution)
        illness_list = []

        for i in range(9):
            if total_distribution[i] > 0.35:
                illness_list.append(self.label_encoder[i])

        return {'health_rate': health_rate * 100,
                'illness_list': illness_list,
                'recommendations': self.get_recommendations(illness_list)}

    def get_description(self, mark):
        if mark == 'chem':
            return None
        elif '-' in mark:
            measure, action = mark.split('-')
            return "{act} {mes}".format(act='Увеличить' if action == 'hi' else 'уменьшить',
                                        mes=self.measure_dict[measure])
        else:
            return "Бороться с {}".format('насекомыми' if mark == 'insects' else 'сорняками')

    def get_recommendations(self, illness_list: List[str]) -> List[Dict[str, str]]:
        conclusion = {}
        for disease in illness_list:
            recommend_dict = self.create_recommendation(disease)
            for key in recommend_dict:
                if key not in conclusion:
                    conclusion[key] = set([])
                conclusion[key] |= recommend_dict[key]
        final_keys = [key for key in conclusion if len(conclusion[key]) == 1 or key == 'chem']
        report = []
        for key in final_keys:
            flag = key + '-' + list(conclusion[key])[0] if type(
                list(conclusion[key])[0]) == str and key != 'chem' else key
            description = self.get_description(flag)
            if description is None: description = "Применить {}".format(', '.join(list(conclusion[flag])))
            report.append({
                'type': flag,
                'description': description
            })
        return report

    def create_recommendation(self, disease_name: str) -> Dict[str, Set[str]]:
        guid_keys = self.cure_cookbook[disease_name]
        recommendation = {}
        for guid_key in guid_keys:
            if guid_key == 'chem':
                recommendation[guid_key] = self.alchemy[disease_name]
            elif '-' not in guid_key:
                recommendation[guid_key] = {True}
            else:
                key, val = guid_key.split('-')
                recommendation[key] = {val}
        return recommendation

    def is_healthy(self, photo: np.array) -> bool:
        return bool(self.healthcheck(photo).argmax())

    def get_health_distribution(self, photo) -> np.array:
        return nn.functional.softmax(self.disease_classifier(photo)[0].detach().unsqueeze(0), dim=-1)[0].numpy()

    def set_healthcheck(self, binary_classifier_path):
        self.healthcheck = models.densenet121(pretrained=True)
        num_ftrs = self.healthcheck.classifier.in_features
        self.healthcheck.classifier = nn.Linear(num_ftrs, 2)
        self.healthcheck.load_state_dict(torch.load(binary_classifier_path, map_location=torch.device('cpu')))
        self.healthcheck.eval()

    def set_diseases_classifier(self, diseases_classifier_path):
        self.disease_classifier = models.densenet121(pretrained=True)
        num_ftrs = self.disease_classifier.classifier.in_features
        self.disease_classifier.classifier = nn.Linear(num_ftrs, 9)
        self.disease_classifier.load_state_dict(torch.load(diseases_classifier_path, map_location=torch.device('cpu')))
        self.disease_classifier.eval()


'''

USAGE EXAMPLE

Dr = PlantDoctor()
print(Dr.diagnose('3.jpeg'))

'''
