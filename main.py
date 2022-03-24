import cv2
import torch
from model import SiameseNetwork
from torchvision import transforms as T
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable


MODEL_PATH = 'models/faceVerification_siameses_network.pth'
map_location = 'cpu'
model = SiameseNetwork()
model.load_state_dict(torch.load(MODEL_PATH, map_location=map_location))
model.eval()

IMAGE_FOLDER = 'images'

face_cascade = cv2.CascadeClassifier(
    'models/haarcascade_frontalface_default.xml'
)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

tfms = T.Compose([
    T.Resize((100, 100)),
    T.ToTensor()
])


cap = cv2.VideoCapture(0)

cnt = 0
while True:

    success, img1 = cap.read()
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img1, 1.3, 5)
    text = ""

    if cnt < 100:
        pass
    else:
        img_detection = img1.copy()
        import glob
        for fol in glob.glob(IMAGE_FOLDER+'/*'):
            for fil1 in glob.glob(fol+'/*'):
                img0 = Image.open(fil1)
                img0 = img0.convert("L")
                process_img1 = Variable(tfms(img0))

                faces = face_cascade.detectMultiScale(img1, 1.3, 5)

                try:
                    x, y, w, h = faces[0]
                    process_img2 = Variable(
                        tfms(Image.fromarray(img1[x-30:x-30+w, y+30:y+30+h])))

                    pred1, pred2 = model(process_img1.unsqueeze(0),
                                         process_img2.unsqueeze(0))
                    euclidean_distance = F.pairwise_distance(
                        pred1, pred2, keepdim=True)

                    print(euclidean_distance)
                    if euclidean_distance[0] < 0.9:
                        print(fil1, '------------------------------')
                        cnt = 0
                        text = fil1.split('\\')[1]
                        cv2.rectangle(img_detection, (x-30, y-30),
                                      (x+w+30, y+h+30), (0, 255, 255), 2)
                        cv2.putText(img_detection, 'Face Detected :'
                                    + fil1.split('\\')[1],
                                    (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 0, 25), 1, cv2.LINE_AA)
                        break

                except Exception as e:
                    print(e)
                    pass

    cnt += 1
    try:
        cv2.imshow('Image', img_detection)
    except Exception as e:
        print('The face is not detected ,', e)
        cv2.imshow('Image', img1)

    cv2.waitKey(1)
