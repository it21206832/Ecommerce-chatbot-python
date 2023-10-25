import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import mysql.connector
from keras.models import load_model
import json
import random

#database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="ecombot"
)

if db.is_connected():
    print("Connected to the database")

model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


def handle_product_inquiry(product_name):
    cursor = db.cursor()
    try:
        cursor.execute("SELECT * FROM products WHERE name = %s", (product_name,))
        product = cursor.fetchone()

        if product:
            product_id, name, description, price, availability, promo_info  = product
            response = f"{name}\n - {description}\n - Price: ${price}\n"

            # Check for availability and promotions
            if availability:
                response += " - Availability: In Stock\n"
            else:
                response += " - Availability: Out of Stock\n"

            if promo_info:
                response += f" - Promotions: {promo_info}\n"
            else:
                response += " - No promotions for this laptop\n"

        else:
            response = "I couldn't find information about that product. Please try again."
    except Exception as e:
        response = f"An error occurred while retrieving the information: {str(e)}"
    finally:
        cursor.close()

    return response


# Get order details by order id

def get_order_by_id(order_id):

    cursor = db.cursor()
    query = "SELECT * FROM orders WHERE order_id = %s"

    # cursor.execute(query, (order_id,))
    cursor.execute("SELECT * FROM orders WHERE order_id = %s", (order_id,))
    order_data = cursor.fetchone()

    # Fetch and print the results
    cursor.close()

    return order_data



def chatbot_response(msg):

    res = ""  # Initialize the res variable with an empty string
    ints = predict_class(msg, model)
    
    if ints[0]['intent'] == 'product_info':
        product_name = msg  # Assuming that the product name is the same as the user's input
        stopwords = ['laptop', 'can', 'you', 'tell', 'me', 'about', 'product','information','details','I','wanna','know','about','info','give','please']
        querywords = msg.split()
        resultwords  = [word for word in querywords if word.lower() not in stopwords]
        product_name = ' '.join(resultwords)
        res = "Your requested laptop information\n\n" + handle_product_inquiry(product_name) + "\n\nContact us for more information.Our email address ai_minds@gmail.com"
        #res = product_name 

    elif msg.startswith("get order by") or msg.startswith("my order id is ") or msg.startswith("my order id is "):
        order_id = msg.split()[-1]
        order_data = get_order_by_id(order_id)
        if order_data:
            res = f"Order ID: {order_data[0]}\n, Product id: {order_data[1]}\n, Quantity: {order_data[2]}\n, Total prise: {order_data[3]}\n"
        else:
            res = "Order not found."
    
    else:
        # If it's not a specific intent, use the previous code to get a response
        res = getResponse(ints, intents)
    
    return res


