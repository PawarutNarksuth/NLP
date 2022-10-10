# import re
# import pandas as pd
# import matplotlib.pyplot as plt
# import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')

# #df = pd.read_excel (r'Path where the Excel file is stored\File name.xlsx')
# df = pd.read_excel (r".\\sentiment\\TeamHealthRawDataForDemo.xlsx")

# # adding an row_id field to the dataframe, which will be useful for joining later
# df["row_id"] = df.index + 1

# #print first 10 rows
# #print (df.head(10))

# #create a new data frame with "id" and "comment" fields
# df_subset = df[['row_id', 'Response']].copy()

# #data clean-up
# #remove all non-aphabet characters
# df_subset['Response'] = df_subset['Response'].str.replace("[^a-zA-Z#]", " ")

# #covert to lower-case
# df_subset['Response'] = df_subset['Response'].str.casefold()

# #print (df_subset.head(10))

# # set up empty dataframe for staging output
# df1 = pd.DataFrame()
# df1['row_id'] = ['99999999999']
# df1['sentiment_type']='NA999NA'
# df1['sentiment_score']=0

# print('Processing sentiment analysis...')
# sid = SentimentIntensityAnalyzer()
# t_df = df1

# for index, row in df_subset.iterrows():

#     scores = sid.polarity_scores(row[1])
    
#     for key, value in scores.items():
#         temp = [key,value,row[0]]
#         df1['row_id']=row[0]
#         df1['sentiment_type']=key
#         df1['sentiment_score']=value
#         t_df=pd.concat([t_df,df1])

# #remove dummy row with row_id = 99999999999
# t_df_cleaned = t_df[t_df.row_id != '99999999999']

# #remove duplicates if any exist
# t_df_cleaned = t_df_cleaned.drop_duplicates()

# # only keep rows where sentiment_type = compound
# t_df_cleaned = t_df[t_df.sentiment_type == 'compound']
# print(t_df_cleaned.head(10))

# #merge dataframes
# df_output = pd.merge(df, t_df_cleaned, on='row_id' , how='inner')
# print(df_output)

# -------------------------------------
from textblob import TextBlob

def input_textblob(text):
    Intext = TextBlob(text)

    if Intext.polarity > 0:
        return "Positive"
    elif Intext.polarity == 0:
        return "Neutral"
    elif Intext.polarity < 0:
        return "Negative"

#print(input_textblob('you aer so good '))

