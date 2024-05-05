from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import os
import warnings

right = 0
wrong = 0
no_sentiment = 0
end = 4
start = 0
help_n = 2
imdb_data=pd.read_csv('IMDB Dataset.csv')

answers = end - start

imdb_data = imdb_data.sample(frac=1, random_state=1).reset_index(drop=True)

test = imdb_data[start:end].reset_index(drop=True)

testX = test.review

testY = test.sentiment

help = imdb_data[end:].reset_index(drop=True)

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")


dataframe = pd.DataFrame(columns=['review', 'sentiment'])


for i, x in enumerate(testX):
    prompt = ""

    help_list = []

    for j in range(help_n):
        help_list.append(help.sample(n=1).reset_index(drop=True))
        prompt = prompt + help_list[j].review[0] + " Sentiment: " + help_list[j].sentiment[0] + ".\n\n"
        print(len(help_list))
    prompt = prompt + x + " Sentiment: \n\n\n"
    print(prompt)
        

    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")

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
        print("Correct: " + testY[i] + "Wrong: " + result)
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
