import time
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import os
import warnings

time1 = time.time()

right = [0, 0, 0, 0, 0, 0]
wrong = [0, 0, 0, 0, 0, 0]
no_sentiment = [0, 0, 0, 0, 0, 0]
neutral = [0, 0, 0, 0, 0, 0]
end = 1000
start = 0
help_n = 1
imdb_data=pd.read_csv('IMDB Dataset.csv')
accuracy = [0, 0, 0, 0, 0, 0]
wrong_percent = [0, 0, 0, 0, 0, 0]
no_sentiment_percent = [0, 0, 0, 0, 0, 0]
neutral_percent = [0, 0, 0, 0, 0, 0]

answers = end - start

imdb_data = imdb_data.sample(frac=1, random_state=1).reset_index(drop=True)

test = imdb_data[start:end].reset_index(drop=True)

testX = test.review

testY = test.sentiment

help = imdb_data[end:].reset_index(drop=True)

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")


dataframe = pd.DataFrame(columns=['review', 'sentiment'])


for k in range(1):
    
    for i, x in enumerate(testX):
        prompt = ""

        help_list = []

        for j in range(help_n):
            help_list.append(help.sample(n=1).reset_index(drop=True))
            prompt = prompt + "TEXT: " + help_list[j].review[0] + "\nSENTIMENT: " + help_list[j].sentiment[0] + "\n\n"
        prompt = prompt + "TEXT: " + x + "\nSENTIMENT: " 
        input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")

        outputs = model.generate(input_ids, max_new_tokens=50)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if(testY[i] in result.lower()):
            print(i, end=": RIGHT\n")
            print(result)
            right[k] += 1
        elif((testY[i] == "positive" and "negative" in result.lower()) or (testY[i] == "negative" and "positive" in result.lower())):
            print(i, end=": WRONG\n")
            print("Correct: " + testY[i] + " Wrong: " + result)
            wrong[k] += 1
        elif(("DISAPPOINTED" in result or "NEUTRAL" in result)):
            print(i, end=": NEUTRAL\n")
            print(result)
            neutral[k] +=1
        else:
            print(i, end=": NO_SENTIMENT\n")
            print(result)
            no_sentiment[k] += 1

    accuracy[k] = (right[k]/answers) * 100

    wrong_percent[k] = (wrong[k]/answers) * 100

    no_sentiment_percent[k] = (no_sentiment[k]/answers) * 100

    neutral_percent[k] = (neutral[k]/answers) * 100

for k in range(1):
    print("\nK:", k)
    print("Accuracy:", accuracy[k], end="%\n")
    print("Wrong percent:", wrong_percent[k], end="%\n")
    print("No sentiment percent:", no_sentiment_percent[k], end="%\n")
    print("Neutral percent:", neutral_percent[k], end="%\n")
    print("Right: ", right[k])
    print("Wrong: ", wrong[k])
    print("No sentiment: ", no_sentiment[k])
    print("Neutral: ", neutral[k])

time2 = time.time()

print(time2-time1)

print("Spaced", help_n)