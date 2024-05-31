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
start = 0
imdb_data=pd.read_csv('IMDB Dataset.csv')
accuracy = 0
wrong_percent = 0
no_sentiment_percent = 0
neutral_percent = 0


data = pd.read_csv('IMDb/data/test_contrast.tsv', sep='\t')
data = data.sample(frac=1, random_state=1).reset_index(drop=True)
data['Index'] = data.index #creo questa colonna perche piu' avanti non sara' piu possibile accedere agli indici

end = len(data.index)

answers = end - start

testX = data.Text
testY = data.Sentiment

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")


help = []

for help_n in range(4):

    for k in range(6):
        for i, x in enumerate(testX):
            help_data = data.copy() #Copio l'originale per eliminare gli aiuti gia' utilizzati per questo campione
            if(help_n > 0):
                prompt = "Classify the text into negative or positive. Here you have some examples:\n\n"
            else:
                prompt = "Classify the text into negative or positive.\n"
            help_list = []

            for j in range(help_n):
                
                
                help = data.iloc[i] #forzo il programma ad entrare le ciclo while ed eseguire la prima pescata di aiuti
                

                while(help.Text == testX[i]):
                    help = help_data.sample(n=1).iloc[0]
                
                help_list.append(help)
                help_data = help_data.drop(help.Index)
                prompt = prompt + "Example: " + help_list[j].Text + "\nSentiment: " + help_list[j].Sentiment[0] + "\n\n"
            prompt = prompt + "Text: " + x + "\nSentiment: "

            input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")

            outputs = model.generate(input_ids, max_new_tokens=50)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if(testY[i].lower() in result.lower()):
                # print(i, end=": RIGHT\n")
                # print(result)
                right += 1
            elif((testY[i].lower() == "positive" and "negative" in result.lower()) or (testY[i].lower() == "negative" and "positive" in result.lower())):
                # print(i, end=": WRONG\n")
                # print("Correct: " + testY[i] + " Wrong: " + result)
                wrong += 1
            elif(("DISAPPOINTED" in result or "NEUTRAL" in result)):
                # print(i, end=": NEUTRAL\n")
                # print(result)
                neutral +=1
            else:
                # print(i, end=": NO_SENTIMENT\n")
                # print(result)
                # print(testY[i])
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

os.system("paplay complete.wav")

