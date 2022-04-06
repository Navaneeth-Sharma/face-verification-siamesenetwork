import glob
import torch
from model import SiameseNetwork
from torchvision import transforms as T
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

IMAGE_FOLDER = '../images'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tfms = T.Compose([
    mtcnn
])


embeddings = {}

for fol in glob.glob(IMAGE_FOLDER+'/*'):
    try:
        for fil1 in glob.glob(fol+'/*'):
            img0 = Image.open(fil1).convert('RGB')
            process_img1 = Variable(tfms(img0))

            embeddings[fil1] = model(
                process_img1.unsqueeze(0)).detach().numpy()
    except FileNotFoundError:
        print("That folder doesn't exist")


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


# print(embeddings)
with open('../embeddings.json', 'w+') as f:
    json.dump(embeddings, f, cls=NumpyEncoder)
