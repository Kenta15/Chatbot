from django.shortcuts import render
from django.http import JsonResponse

import random
import json
import torch
from .training.model import NeuralNet
from .training.nltk_utils import bag_of_words, tokenize

import speech_recognition as sr
import playsound
import os
import sys
from gtts import gTTS 


# Create your views here.

def home(request):
    context = {}
    return render(request, 'home.html', context)

def alex(request):
    
    def speech():
        r = sr.Recognizer()

        with sr.Microphone() as source:
            audio = r.listen(source)
            try:
                voice_data = r.recognize_google(audio)
                return respond(voice_data)
            except sr.UnknownValueError:
                print('bot: Sorry, I did not get that')
            except sr.RequestError:
                print('bot: Sorry, the service is down')

    def speak(audio_string):
        tts = gTTS(text=audio_string, lang='en')
        r = random.randint(1, 100000000)
        audio_file = 'audio-' + str(r) + '.mp3'
        tts.save(audio_file)
        playsound.playsound(audio_file)
        os.remove(audio_file)

    def respond(voice_data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(os.path.join(sys.path[0],'home/training/intents.json'), 'r') as json_data:
            intents = json.load(json_data)
        FILE = os.path.join(sys.path[0],'home/training/data.pth')
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

        global raw_sentence
        raw_sentence = voice_data
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
                    global response
                    response = random.choice(intent['responses'])
                    print(f"{bot_name}: {response}")
                    speak(response)
        else:
            print(f"{bot_name}: I do not understand.. ")
            response = 'I do not understand..'
            speak('I do not understand')
    
    speech()
    return JsonResponse({'input': raw_sentence, 'response': response})
    

def chatbot(request):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(sys.path[0],'home/training/intents.json'), 'r') as json_data:
        intents = json.load(json_data)
    FILE = os.path.join(sys.path[0],'home/training/data.pth')
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

    # getting fetching input from frontend
    json_input = json.loads(request.body)
    print(json_input)
    input = json_input['input'].lower()
    
    sentence = input

    if sentence =='q':
        return JsonResponse({'response': 'Bye!'})

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    print(probs)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                response = random.choice(intent['responses'])
    else:
        response = 'Oops! Sorry, I do not understand...🤖'

    return JsonResponse({'response': response})