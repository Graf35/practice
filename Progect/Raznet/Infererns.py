import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms

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
        x = nn.AdaptiveAvgPool2d((1 ,1))(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x), self.bb(x)

def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color=color,
                         fill=False, lw=3)
def show_corner_bb(im, bb):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))

model = BB_model().cuda()
model.load_state_dict(torch.load("model.pth"))
model.eval()

path="images/road96.png"

im = cv2.imread(path)
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
# predicted bounding box
bb_hat = out_bb.detach().cpu().numpy()
bb_hat = bb_hat.astype(int)
show_corner_bb(im, bb_hat[0])
class_dict = { 0:'speedlimit', 1: 'stop', 2: 'crosswalk', 3: 'trafficlight'}
print(class_dict[int(out_class.argmax())])