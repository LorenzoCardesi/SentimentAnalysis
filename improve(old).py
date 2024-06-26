from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import os
import warnings

right = 0
wrong = 0
no_sentiment = 0
end = 1000
start = 0
help = 3
imdb_data=pd.read_csv('IMDB Dataset.csv')


imdb_data=imdb_data.sample(n=end, random_state=1).reset_index(drop=True)

testX = imdb_data.review[start:end].reset_index(drop=True)

testY = imdb_data.sentiment[start:end].reset_index(drop=True)

answers = end - start - help

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")

prompt = ""

print(testY[0:10])
for i, x in enumerate(testX):
    if(i<help):
        prompt = prompt + x + " Sentiment: " + testY[i] + "\n"
        print(prompt)

    else:
        test = prompt + x + " Sentiment:"
        input_ids = tokenizer(test, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")

        outputs = model.generate(input_ids, max_new_tokens=50)
        result = tokenizer.decode(outputs[0])

        ignored_token = ["<pad> ", "</s>"]
        for token in ignored_token:
            result = result.replace(token, "")

        if(testY[i] == result):
            print(i, end=": RIGHT\n")
            right += 1
        elif(result.endswith("Sentiment: " + testY[i])):
            print(i, end=": RIGHT\n")
            print(result)
            right += 1
        elif((testY[i] == "positive" and result.endswith("negative")) or (testY[i] == "negative" and result.endswith("positive"))):
            print(i, end=": WRONG\n")
            print(result)
            wrong += 1
        else:
            print(i, end=": NO_SENTIMENT\n")
            print(result)
            no_sentiment += 1

accuracy = (right/answers) * 100

wrong_percent = (wrong/answers) * 100

no_sentiment_percent = (no_sentiment/answers) * 100

print("Accuracy:", accuracy, end="%\n")
print("Wrong percent:", wrong_percent, end="%\n")
print("No sentiment percent:", no_sentiment_percent, end="%\n")


print("Right:", right)
print("Wrong:", wrong)
print("No sentiment: ", no_sentiment)
