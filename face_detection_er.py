# Perform face detection cropping and emotion recognition

# !pip install mtcnn
# !pip install facenet-pytorch


import numpy as np
from mtcnn import MTCNN as m_mtcnn
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import transforms as transforms
from models.vgg import VGG
import cv2

class FaceFeatures:

    def __init__(self):

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        cut_size = 44
        self.transform = transforms.Compose([
            transforms.TenCrop(cut_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])
        # checkpoint = torch.load(os.path.join("./FER2013_VGG19", "PrivateTest_model.t7"), map_location = self.device)
        checkpoint = torch.load('/content/multimodal_feat_extraction_app/FER2013_VGG19/PrivateTest_model.t7',map_location=self.device)
        
        self.fer_model = VGG('VGG19')
        self.fer_model.load_state_dict(checkpoint['net'])
        self.fer_model.to(self.device)
        self.face_detector_model = m_mtcnn()


    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


    def get_face_output(self, frame):

        prediction = self.face_detector_model.detect_faces(frame)

        return prediction
    
    def crop_faces(self, frame, frame_face_pred):
        
        crop_face_list = []
        for face in frame_face_pred:
            x, y, w, h = face["box"][0], face["box"][1], face["box"][2], face["box"][3] # crop the detected face\
            cropped_face = frame[y:y+h, x:x+w]
            # convert to grayscale
            cropped_face = self.rgb2gray(cropped_face)
            # resized according to the fer2013 dataset
            cropped_face = cv2.resize(cropped_face, (48, 48))
            crop_face_list.append(cropped_face)

        return crop_face_list

    def get_emotion(self, crop_face_list):

        frame_emotion = np.zeros(7)
        for cropped_face in crop_face_list:
            
            # Preprocessing to remove color bias
            img = cropped_face[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis = 2)
            img = Image.fromarray(img.astype(np.uint8))
            inputs = self.transform(img)

            ncrops, c, h, w = np.shape(inputs)
            inputs = inputs.view(-1, c, h, w)
            with torch.no_grad():
                inputs = inputs.to(self.device)

            fer_logits = self.fer_model(inputs) # face emotion recognition
            fer_logits = fer_logits.view(ncrops, -1).mean(0)
            score = F.softmax(fer_logits, dim=0)
            frame_emotion += score.cpu().detach().numpy()
        
        frame_emotion/=len(crop_face_list)

        return frame_emotion
