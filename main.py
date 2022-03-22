import cv2
import torch
from faceVerifucationModel import SiameseModel2
from torchvision import transforms as T
import torch.nn.functional as F
import glob


cap = cv2.VideoCapture(0)
MODEL_PATH = 'models/faceVerification_siameses_Model_2__.pth'
map_location = 'cpu'

model = SiameseModel2()
model.load_state_dict(torch.load(MODEL_PATH, map_location=map_location))
model.eval()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

tfms = T.Compose([
    T.ToPILImage(),
    T.Resize(224),
    # T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
])


def preprocess_image(img, IMAGE_SIZE=224):
    im = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    im = tfms(im)
    return im[None]


def process_data(imgs1, imgs2):
    imgs1 = preprocess_image(imgs1)
    imgs2 = preprocess_image(imgs2)
    return imgs1, imgs2




cnt = 0
string = " Not found"
while True:
    success, img = cap.read()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    if cnt<100:
        
        pass
    else:
        string = "Not found"
        for fol in glob.glob('images/*'):
            for fil in glob.glob(fol+'/*'):
                img_verifiy = cv2.imread(fil)
                img_verifiy = cv2.cvtColor(img_verifiy, cv2.COLOR_BGR2RGB)

                process_img1, process_img2 = process_data(img_verifiy, img)

                pred1, pred2 = model(process_img1, process_img2)
                euclidean_distance = F.pairwise_distance(pred1, pred2, keepdim=True)

                if euclidean_distance[0][0]<0.6:
                    string = fol.split('\\')[-1] + " your attendence is marked!"
                    cnt = 0
                    break

    cnt+=1
    
    org1 = (20, 20)
    thickness1 = 2
    font1 = cv2.FONT_HERSHEY_SIMPLEX
    color1 = (0, 255, 0)
    fontScale1 = 1

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", img)

    img1 = cv2.putText(img, string,
                    org1, font1, fontScale1, color1, thickness1, cv2.LINE_AA)
    cv2.imshow('Image', img1)

    cv2.waitKey(1)
