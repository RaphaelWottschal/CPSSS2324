import ipywidgets.widgets as widgets
import logging
import os
import cv2
import torchvision
import torch
import sys
import time
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image
import numpy as np
from IPython.display import display
import ipywidgets
import traitlets
import socket
import zmq
import pickle


# Laden des Modells
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
print("Loading model")
model.load_state_dict(torch.load('2best_steering_model_xy.pth', map_location=torch.device('cpu')))
#device = torch.device('cuda')
print("Model loaded")
device = torch.device('cpu')

model = model.to(device).eval()

mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])

server_ip = '10.11.25.26'

def preprocess(image, device, mean, std):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def run_detection_loop():
    context = zmq.Context()

    server_ip = "10.11.25.26"

    # Receiver Socket konfigurieren (zum Empfang von Frames)
    receiver_socket = context.socket(zmq.PULL)
    receiver_socket.setsockopt(zmq.CONFLATE, 1)  # Ältere Nachrichten verwerfen
    receiver_socket.connect(f"tcp://{server_ip}:5555")  # IP-Adresse des Servers

    # Sender Socket konfigurieren (zum Senden der Ergebnisse)
    sender_socket = context.socket(zmq.PUSH)
    sender_socket.connect(f"tcp://{server_ip}:5557")  # IP-Adresse des Servers

    while True:
        # Frame von ZeroMQ empfangen
        frame_bytes = receiver_socket.recv()

        # JPEG-Frame decodieren
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Bild in RGB umwandeln und Größe anpassen
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (224, 224))  # Stellen Sie sicher, dass die Größe konsistent ist

        xy = model(preprocess(resized_frame, device, mean, std)).detach().float().cpu().numpy().flatten()

        print(xy)

        sender_socket.send(pickle.dumps([1, xy]))

if __name__ == "__main__":
    print(run_detection_loop())