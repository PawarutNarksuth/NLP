from transformers import BertTokenizerFast, BertForSequenceClassification
import pandas as pd

#fake-news-dataset-mee-model

model_name = "bert-base-uncased"

# max sequence length
max_length = 512

# load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

# load the model
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

def get_prediction(text, convert_to_label=False):
    inputs = tokenizer(text, padding=True, truncation=True, max_length = max_length, return_tensors="pt")
    
    outputs = model(**inputs)
        
    probs = outputs[0].softmax(1)
        
    d = {
        0: "reliable",
        1: "fake"
    }
        
    if convert_to_label:
        return d[int(probs.argmax())]
    else:
        return int(probs.argmax())

real_news = """ Tim Tebow Will Attempt Another Comeback, This Time in Baseball -
            The New York Times",Daniel Victor,"If at first you don’t
            succeed, try a different sport. Tim Tebow, who was a
            Heisman quarterback at the University of Florida but was
            unable to hold an N. F. L. job, is pursuing a career in Major
            League Baseball. <SNIPPED>
            """

# print(get_prediction(real_news, convert_to_label=True))
