# GUI Packages
import tkinter
from tkinter import *
# package for manipulating arrays
import numpy
# machine learning packages which should be used later on
import tflearn
import tensorflow
import random
# how the intents file is being imported for use
import json
# package for natural language like turning words in their base form
# like what's to what
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import pickle
from tkinter.ttk import *

with open("intents.json") as file:
    data = json.load(file)
# prints out entire set of dictionaries
try:
    with open("data.pickle", "rb") as f:
        processed_words, groups, trainer, output = pickle.load(f)
except:
    processed_words = []
    groups = []
    responses = []
    response_group = []
    
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            words = nltk.word_tokenize(pattern)
            processed_words.extend(words)
            responses.append(words)
            response_group.append(intent["group"])
        if intent["group"] not in groups:
            groups.append(intent["group"])
    
    processed_words = [stemmer.stem(w.lower()) for w in processed_words if w != "?"]
    processed_words = sorted(list(set(processed_words)))
    
    groups = sorted(groups)
    
    trainer = []
    output = []
    
    output_empty = [0 for _ in range(len(groups))]
    for x, doc in enumerate(responses):
        word_list = []
    
        words = [stemmer.stem(w) for w in doc]
    
        for w in processed_words:
            if w in words:
                word_list.append(1)
            else:
                word_list.append(0)
    
        output_row = output_empty[:]
        output_row[groups.index(response_group[x])] = 1
    
        trainer.append(word_list)
        output.append(output_row)
    
    trainer = numpy.array(trainer)
    output = numpy.array(output)
    with open("data.pickle", "wb") as f:
            pickle.dump((processed_words, groups, trainer, output), f)

tensorflow.reset_default_graph()

ai = tflearn.input_data(shape=[None, len(trainer[0])])
ai = tflearn.fully_connected(ai, 8)
ai = tflearn.fully_connected(ai, 8)
ai = tflearn.fully_connected(ai, len(output[0]), activation="softmax")
ai = tflearn.regression(ai)

model = tflearn.DNN(ai)
try:
    model.load("model.tflearn")
except:
    model = tflearn.DNN(ai)
    model.fit(trainer, output, n_epoch=1000, batch_size=8, show_metric=False)
    model.save("model.tflearn")


def sentence_in(a, processed_words):
    sentence = [0 for _ in range(len(processed_words))]
    sen_processed = nltk.word_tokenize(a)
    sen_processed = [stemmer.stem(word.lower()) for word in sen_processed]
    for x in sen_processed:
        for i, w in enumerate(processed_words):
            if w == x:
                sentence[i] = 1
    return numpy.array(sentence)


def chat():
    # while True:
    prompt1 = EnterSection.get("1.0", 'end-1c').strip()
    EnterSection.delete("0.0", tkinter.END)
    ChatSection.insert(tkinter.END, "you: " + prompt1 + '\n\n', "b")
    # prompt = input("you:")
    # if prompt.lower() == "exit":
    # print("Closing chatbot!")
    # break
    #print(prompt1)
    answers = model.predict([sentence_in(prompt1, processed_words)])[0]
    answers_index = numpy.argmax(answers)
    head = groups[answers_index]

    if answers[answers_index] > 0.7:
        for header in data["intents"]:
            if header['group'] == head:
                responses = header['responses']
        text = random.choice(responses)
        ChatSection.insert(tkinter.END, "Advising Bot: " +
                        text + '\n\n',"a")
    else:
        ChatSection.insert(tkinter.END, "Advising Bot: I didn't understand that, can you rephrase?" + '\n\n', "a")
        # miss = "I didn't understand that, can you rephrase?"
        # return miss


window = tkinter.Tk()
window.title("CSUDH Advising Bot")
window.geometry("500x530")
ChatSection = tkinter.Text(window, height="10", width="40", bd=3)
# ChatSection.config(state=DISABLED)
Scroll = tkinter.Scrollbar(window, command=ChatSection.yview)
ChatSection['yscrollcommand'] = Scroll.set
ChatSection.tag_config("a", foreground="blue", wrap = WORD)
ChatSection.tag_config("b", wrap = WORD)
ChatSection.insert(tkinter.END, "Hello! I am the advising Chat Bot, How can I help you?" + '\n\n')
EnterButton = tkinter.Button(window, text="Enter", command=chat)
EnterSection = tkinter.Text(window, bd=3, bg="white", width="30",
                            height="10")
Scroll.place(x=476, y=6, height=486)
ChatSection.place(x=10, y=6, height=486, width=470)
EnterSection.place(x=10, y=497, height=25, width=450)
EnterButton.place(x=460, y=495, height=25)

# chat()
window.mainloop()
