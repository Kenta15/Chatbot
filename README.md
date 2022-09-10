# Chatbot

# Chatbot
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Django](https://img.shields.io/badge/django-%23092E20.svg?style=for-the-badge&logo=django&logoColor=white)
![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## Overview

What I learned:

* How a chatbot works generally using tokenization, stemming, and a bag of words
* How to use a neural network to make a deep learning chatbot

## How a chatbot works

### Tokenize
• Split a sentence into a word

`'how are you' -> ['how', 'are', 'you']
`

### Stemming
• reduce words into its root

`['delivery', 'delivered', 'delivering'] -> ['deliv', 'deliv', 'deliv]
`

### Bag of words
• counts occurence of words

`input: how are you
['how', 'is', 'it', 'I', 'am', 'are', 'sell, 'you'] -> [1, 0, 0, 0, 0, 1, 0, 1]
`
### Result

Then, it trains our model with a nural network. 

After all, the bot chooses one of the corresponding responses even though the exact input is not there.

```
{
  "tag": "greeting",
  "patterns": [
      "Hi",
      "Hello",
      "What is up",
      "How are you doing",
      "Hey",
      "yo",
      "sup"
  ],
  "responses":[
      "Hey!",
      "Hello",
      "Hi there, how can I help?",
      "Hi, what can I do for you today?",
      "Hi, there"
  ]
}
```
        
You: How are you?

Bot: Hi, there

## Setup

Download [Python](https://www.python.org/downloads/).
Run these commands:

```diff
# Install Django
python -m pip install Django

# Move to the file directory
cd movie_recommendation

# Run the local server
python manage.py runserver
```

## Demo

Video Link: 
https://user-images.githubusercontent.com/92241890/189480427-fa3177ce-5785-4d16-82d6-ba43220c73bf.mp4


## References
Followed this tutorial: [Training our model](https://www.youtube.com/watch?v=RpWeNzfSUHw), [Speech to Text](https://www.youtube.com/watch?v=x8xjj6cR9Nc&t=73s)
