import torch
import clip
from PIL import Image
import cv2
from torchvision.transforms import Compose, Resize, InterpolationMode, Normalize

import numpy as np

import os
from glob import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

file_list = glob('output/RESNET_150/*')
file_list = sorted(file_list)
my_preprocess = Compose([Resize(224, interpolation=InterpolationMode.BICUBIC),
                        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                        ])

score = np.zeros(shape=(1,9))

i = 0
for f in file_list:
    with torch.no_grad():
        prompt = f.split('/')[-1].split('.')[0]
        images = glob(os.path.join('corpus', prompt)+'/*')
        images = sorted(images)

        out = torch.randint(low=0, high=255, size=(255,255,3))

        x = list()
        for image in images:
            img = cv2.imread(image)
            x.append(img)
        if(len(x) == 0):
            continue

        i += 1
        x = np.array(x)

        out = torch.tensor(out, dtype=float)
        out = torch.unsqueeze(out, dim=0).permute(0,3,1,2)
        x = torch.tensor(x, dtype=float)
        x = torch.squeeze(x).permute(0,3,1,2)

        x = torch.cat([my_preprocess(out), my_preprocess(x)], dim=0)

        text = torch.cat([clip.tokenize([prompt]).to(device)])
        logits_per_image, logits_per_text = model(x, text)
        logits_per_image = logits_per_image.reshape(1,-1)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        score += probs
        print('score: ', score)

score = score / i
print('i: ', i)
print('total score: ', score)
