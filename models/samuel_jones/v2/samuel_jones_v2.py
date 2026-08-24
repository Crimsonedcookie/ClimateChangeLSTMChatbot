from pymongo.mongo_client import MongoClient
import pymongo
from flask import Flask, render_template, request, redirect
from transformers import pipeline
from datetime import datetime

app=Flask("climatemanager")

uri = "mongodb+srv://nramani27:tort524TOTE@climatechatbot.zizrvai.mongodb.net/?retryWrites=true&w=majority&appName=climatechatbot"
# Create a new client and connect to the server
# client = MongoClient(uri)

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['GET','POST'])
def chatbot():
    if request.method == "GET":
        return render_template('chatbot.html')
    if request.method == "POST":
        model_name = "google/flan-t5-base"

        qa = pipeline(
            "text2text-generation",
            model=model_name,
            device=-1
        )
        # Number of tokens is deciding factor
        def generate_climate_answer(prompt, max_tokens=256):
            result = qa(prompt, max_new_tokens=max_tokens)
            return result[0]['generated_text']
        print(request.form)
        chatbot_output=generate_climate_answer(request.form["question"])
        chatbot_question=request.form["question"]
        print(chatbot_output)
        return render_template("chatbot.html",chatbot_output=chatbot_output,chatbot_question=chatbot_question,date=datetime.now())

@app.route('/sources')
def sources():
    return render_template('sources.html')

app.run(debug=True,port=5001)

#Model Q&A

#Explain global warming.
#Global warming is warming of the Earth's climate due to greenhouse gas emissions

#What causes climate change?
#Climate change is driven by the emission of greenhouse gases into the atmosphere

#Explain the ozone layer?
#The ozone layer is made up of O3 molecules (also called Ozone). Ozone is responsible for intercepting heat from the sun before it reaches Earth's surface. However, ozone reacts with Carbon Monoxide to produce Carbon Dioxide. So when Carbon Monoxide (or other similar molecules) are emitted, they react to form a hole in the ozone layer, allowing more heat into the atmosphere and warming up the planet.

#Explain how deforestation impacts wildlife.
#Deforestation eliminates the habitats of wildlife. It also destroys ecosystems vital to the functioning of the interconnected food web, altering the equilibrium.

#Explain the consequences of climate change on marine life.
#The higher concentration of carbon in the atmosphere is regulated by the oceans, which absorb carbon. However, the increased concentration of carbon in the ocean leads to marine life damage.

#
