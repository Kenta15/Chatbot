import speech_recognition as sr
import json
import torch
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
import random

def speech():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        audio = r.listen(source)
        try:
            voice_data = r.recognize_google(audio)
            if voice_data == 'exit':
                exit()
            return respond(voice_data)
        except sr.UnknownValueError:
            print('bot: Sorry, I did not get that')
        except sr.RequestError:
            print('bot: Sorry, the service is down')

def respond(voice_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)
    FILE = 'data.pth'
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

    sentence = voice_data
    print(f"you: {sentence}")

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    bot_name = "bot"

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand.. ")

print('bot: How can I help you? (Say "exit" to exit)')

while True:
    speech()
