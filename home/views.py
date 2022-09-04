from django.shortcuts import render

import random
import json
import torch
from training.model import NeuralNet
from training.nltk_utils import bag_of_words, tokenize



# Create your views here.

def home(request):

    context = {}
    return render(request, 'home.html', context)

device = torch.device('cuba' if torch.cuda.is_available() else 'cpu')
with open('training/intents.json', 'r') as json_data:
    intents = json.load(json_data)
FILE = 'training/data.pth'
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Boty"
print('lets have a chat')

while True:
    sentence = input("You: ")
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand.. ")