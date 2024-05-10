import time
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import os
import warnings

time1 = time.time()

right = 0
wrong = 0
no_sentiment = 0
neutral = 0
end = 1000
start = 0
imdb_data=pd.read_csv('IMDB Dataset.csv')
accuracy = 0
wrong_percent = 0
no_sentiment_percent = 0
neutral_percent = 0

answers = end - start

imdb_data = imdb_data.sample(frac=1, random_state=1).reset_index(drop=True)

test = imdb_data[start:end].reset_index(drop=True)

testX = test.review

testY = test.sentiment

help = imdb_data[end:].reset_index(drop=True)

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")


dataframe = pd.DataFrame(columns=['review', 'sentiment'])

for help_n in range(4, 5):

    for k in range(1):
        
        for i, x in enumerate(testX):
            if(help_n > 0):
                prompt = "Classify the text into negative or positive. Here you have some examples:\n\n"
            else:
                prompt = "Classify the text into negative or positive.\n"
            help_list = []

            for j in range(help_n):
                help_list.append(help.sample(n=1).reset_index(drop=True))
                prompt = prompt + "Example: " + help_list[j].review[0] + "\nSentiment: " + help_list[j].sentiment[0] + "\n\n"
            prompt = prompt + "Text: " + x + "\nSentiment: " 
            input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")

            outputs = model.generate(input_ids, max_new_tokens=50)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if(testY[i] in result.lower()):
                #print(i, end=": RIGHT\n")
                #print(result)
                right += 1
            elif((testY[i] == "positive" and "negative" in result.lower()) or (testY[i] == "negative" and "positive" in result.lower())):
                #print(i, end=": WRONG\n")
                #print("Correct: " + testY[i] + " Wrong: " + result)
                wrong += 1
            elif(("DISAPPOINTED" in result or "NEUTRAL" in result)):
                #print(i, end=": NEUTRAL\n")
                #print(result)
                neutral +=1
            else:
                #print(i, end=": NO_SENTIMENT\n")
                #print(result)
                no_sentiment += 1

        accuracy = (right/answers) * 100

        wrong_percent = (wrong/answers) * 100

        no_sentiment_percent = (no_sentiment/answers) * 100

        neutral_percent = (neutral/answers) * 100
        print("K:", k , " Help:", help_n)
        print(accuracy, wrong_percent, no_sentiment_percent, neutral_percent, end="%\n")

        right = 0
        wrong = 0
        no_sentiment = 0
        neutral = 0

time2 = time.time()

print(time2-time1)

"""
Classify the text into negative or positive. Here you have some examples:

Example: (prompt).
Sentiment: positive             PROMPT1

Text: (prompt).
Sentiment:
"""

"""
Classify the text into negative or positive. Here you have some examples:

Text: (prompt).
Sentiment: positive             PROMPT2

Text: (prompt).
Sentiment:
"""

"""
Classify the Sentiment of the TEXT. You can only choose between NEGATIVE or POSITIVE. Here you have some examples:

TEXT: (prompt).
SENTIMENT: POSITIVE
                                PROMPT3
TEXT: (prompt).
SENTIMENT:
"""

"""
Classify the sentiment of the text. you can only choose between NEGATIVE or POSITIVE. Here you have some examples:

Text: (prompt).
Sentiment: POSITIVE
                                PROMPT4
Text: (prompt).
Sentiment:
"""
