import cv2
import os
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms

def filelist(root, file_type):
    """Функция возвращает полностью квалифицированный список файлов в директории"""
    return [os.path.join(directory_path, f) for directory_path, directory_name,
            files in os.walk(root) for f in files if f.endswith(file_type)]


def generate_train_df (anno_path):
    annotations = filelist(anno_path, '.xml')
    anno_list = []
    for anno_path in annotations:
        root = ET.parse(anno_path).getroot()
        anno = {}
        anno['filename'] = Path(str(images_path) + '/'+ root.find("./filename").text)
        anno['width'] = root.find("./size/width").text
        anno['height'] = root.find("./size/height").text
        anno['class'] = root.find("./object/name").text
        anno['xmin'] = int(root.find("./object/bndbox/xmin").text)
        anno['ymin'] = int(root.find("./object/bndbox/ymin").text)
        anno['xmax'] = int(root.find("./object/bndbox/xmax").text)
        anno['ymax'] = int(root.find("./object/bndbox/ymax").text)
        anno_list.append(anno)
    return pd.DataFrame(anno_list)


def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3] - bb[1], bb[2] - bb[0], color=color,
                         fill=False, lw=3)


def show_corner_bb(im, bb):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))


def accuracybb(df_train, out_bb, i):
    xmin = df_train["xmin"][i]
    ymin = df_train["ymin"][i]
    xmax = df_train["xmax"][i]
    ymax = df_train["ymax"][i]
    bb = out_bb.cpu().detach().numpy().tolist()
    xminbb = bb[0][0]
    yminbb = bb[0][1]
    xmaxbb = bb[0][2]
    ymaxbb = bb[0][3]
    if xminbb > xmax or xmaxbb < xmin or yminbb > ymax or ymaxbb < ymin:
        s_unification = 0
    else:
        x1 = abs(xmin - xminbb)
        x2 = abs(xmax - xmaxbb)
        y1 = abs(ymin - yminbb)
        y2 = abs(ymax - ymaxbb)
        length_y = max([ymax, ymin, yminbb, ymaxbb]) - min([ymax, ymin, yminbb, ymaxbb])
        length_x = max([xmax, xmin, xminbb, xmaxbb]) - min([xmax, xmin, xminbb, xmaxbb])
        a = length_y - y1 - y2
        b = length_x - x1 - x2
        s_intersections = a * b
        s_object = (xmax - xmin) * (ymax - ymin)
        s_unification = s_intersections / s_object
    return s_unification


class BB_model(nn.Module):
    def __init__(self):
        super(BB_model, self).__init__()
        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())[:8]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x), self.bb(x)

images_path = Path('images')
anno_path = Path('annotations')
df_train = generate_train_df(anno_path)
class_dict = {'speedlimit': 0, 'stop': 1, 'crosswalk': 2, 'trafficlight': 3}
df_train['class'] = df_train['class'].apply(lambda x:  class_dict[x])

model = BB_model().cuda()
model.load_state_dict(torch.load("model.pth"))
model.eval()
Tru_class=0
bb_accuracy=0
for i in range(df_train.shape[0]):
    path=df_train["filename"][i]
    im = cv2.imread(str(path))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    transform_object = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    image_tensor = transform_object(im)
    image_tensor = image_tensor.unsqueeze(0)
    out_class, out_bb = model(image_tensor.cuda())
    maxintd=torch.max(out_class, 1)
    if df_train["class"][i]==int(out_class.argmax()):
        Tru_class+=1
    bb_accuracy+=accuracybb(df_train, out_bb, i)

accuracy_bb=bb_accuracy/(i+1)
accuracy_class=Tru_class/df_train.shape[0]*100
print(accuracy_class, accuracy_bb)