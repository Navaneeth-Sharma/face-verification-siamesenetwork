import cv2
import torch
from model import SiameseNetwork
from torchvision import transforms as T
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from facenet_pytorch import MTCNN
import json
import numpy as np


MODEL_PATH = 'models/faceVerification_siameses_network_.pth'
map_location = 'cpu'
model = SiameseNetwork()
model.load_state_dict(torch.load(MODEL_PATH, map_location=map_location))
model.eval()

mtcnn = MTCNN(image_size=224, margin=50)

IMAGE_FOLDER = 'images'

face_cascade = cv2.CascadeClassifier(
    'models/haarcascade_frontalface_default.xml'
)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

tfms = T.Compose([
    mtcnn
])


cap = cv2.VideoCapture(0)

cnt = 0


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


file = open("embeddings.json", 'r')
json_data = json.load(file)

# print(json_data)

while True:

    success, img1 = cap.read()
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(img1, 1.3, 5)
    text = ""

    if cnt < 100:
        pass
    else:
        img_detection = img1.copy()
        for i in json_data:
            pred1 = torch.tensor(json_data[i])

            faces = face_cascade.detectMultiScale(img1, 1.1, 3)

            try:
                x, y, w, h = faces[0]
                process_img2 = Image.fromarray(img1)

                process_img2 = Variable(
                    tfms(process_img2))

                pred2 = model(process_img2.unsqueeze(0))

                euclidean_distance = F.pairwise_distance(
                    pred1, pred2, keepdim=True)

                print(i, euclidean_distance)
                if euclidean_distance[0] < 3:
                    print(i, '------------------------------')
                    cnt = 0
                    text = i.split('\\')[1]
                    # img_detection = img_detection[x-60:x+w+60, y-60:y+h+60]
                    cv2.rectangle(img_detection, (x-60, y-60),
                                  (x+w+60, y+h+60), (0, 255, 255), 2)
                    cv2.putText(img_detection, 'Face Detected :'
                                + i.split('\\')[1],
                                (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 25), 1, cv2.LINE_AA)
                    # break

            except Exception as e:
                print(e)
                pass

    cnt += 1
    try:
        cv2.imshow('Image', cv2.cvtColor(img_detection, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print('The face is not detected ,', e)
        cv2.imshow('Image', cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

    cv2.waitKey(1)
