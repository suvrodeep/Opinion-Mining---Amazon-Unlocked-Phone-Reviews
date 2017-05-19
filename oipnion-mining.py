import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

file = str("\\files\output_complete_file.txt")
initial_data_frame = pd.read_csv(file, header=None, names=["brand", "rating", "topic", "word", "score"])

initial_data_frame["pos_score"] = float(0)
initial_data_frame["neg_score"] = float(0)
initial_data_frame["neu_score"] = float(0)
initial_data_frame["comp_score"] = float(0)

# Defining feature words
feature_words = ["phone", "camera", "price", "battery", "charge", "sound", "memory", "screen", "speaker"]

# Scoring words
scorer = SentimentIntensityAnalyzer()
score = dict()
for row_num in range(0, len(initial_data_frame), 1):
    word = initial_data_frame['word'][row_num]
    if word in feature_words:
        initial_data_frame.set_value(row_num, 'pos_score', 0)
        initial_data_frame.set_value(row_num, 'neg_score', 0)
        initial_data_frame.set_value(row_num, 'neu_score', 0)
        initial_data_frame.set_value(row_num, 'comp_score', 0)
    else:
        score = scorer.polarity_scores(word)
        pos_wtd_score = float(score['pos'] * initial_data_frame['score'][row_num] * 100)
        neg_wtd_score = float(score['neg'] * initial_data_frame['score'][row_num] * 100)
        neu_wtd_score = float(score['neu'] * initial_data_frame['score'][row_num] * 100)
        comp_wtd_score = float(score['compound'] * initial_data_frame['score'][row_num] * 100)

        initial_data_frame.set_value(row_num, 'pos_score', pos_wtd_score)
        initial_data_frame.set_value(row_num, 'neg_score', neg_wtd_score)
        initial_data_frame.set_value(row_num, 'neu_score', neu_wtd_score)
        initial_data_frame.set_value(row_num, 'comp_score', comp_wtd_score)

# Filtering topics
topics = pd.unique(initial_data_frame['topic'])
brands = pd.unique(initial_data_frame['brand'])
final_df = pd.DataFrame(data=initial_data_frame)
for brand in brands:
    for topic in topics:
        temp_df = pd.DataFrame(initial_data_frame[initial_data_frame['brand'] == brand])
        temp_df = temp_df[temp_df['topic'] == topic]
        temp_df = temp_df.reset_index(drop=True)
        flag = 1
        for row_num in range(0, len(temp_df), 1):
            word = temp_df['word'][row_num]
            # print(str(brand) + " : " + str(topic) + " : " + str(word) + " : Before if " + str(flag))

            if word in feature_words:
                flag = 0
                # print(str(brand) + " : " + str(topic) + " : " + str(word) + " : After if " + str(flag))
                break
            else:
                flag = 1

        if flag == 1:
            to_remove = initial_data_frame[
                (initial_data_frame['topic'] == topic) & (initial_data_frame['brand'] == brand)].index.tolist()
            # print(to_remove)
            final_df = final_df.drop(to_remove, axis=0)
        else:
            continue

# Scoring feature words
topics = pd.unique(final_df['topic'])
brands = pd.unique(final_df['brand'])
result = []

for brand in brands:
    for topic in topics:
        temp_df = pd.DataFrame(final_df[final_df['brand'] == brand])
        temp_df = temp_df[temp_df['topic'] == topic]
        temp_df = temp_df.reset_index(drop=True)
        temp_df_wo_ftr = pd.DataFrame(temp_df[~temp_df['word'].isin(feature_words)])
        temp_df_w_ftr = pd.DataFrame(temp_df[temp_df['word'].isin(feature_words)])

        pos = temp_df_wo_ftr['pos_score'].sum()
        neg = temp_df_wo_ftr['neg_score'].sum()
        neu = temp_df_wo_ftr['neu_score'].sum()
        comp = temp_df_wo_ftr['comp_score'].sum()

        for ftr_wrd in feature_words:
            count = temp_df_w_ftr[temp_df_w_ftr['word'] == ftr_wrd].count()[1]
            append_list = [brand, ftr_wrd, count, pos, neg, neu, comp]
            result.append(append_list)

result = pd.DataFrame(data=result,
                      columns=["Brand", "Feature", "Topic Word Density", "Positive", "Negative", "Neutral", "Compound"])

df = pd.DataFrame(data=result.groupby(["Brand", "Feature"]).sum())
df['Positive'] = round(df['Positive'] * df['Topic Word Density'] / 100, 3)
df['Negative'] = round(df['Negative'] * df['Topic Word Density'] / 100, 3)
df['Neutral'] = round(df['Neutral'] * df['Topic Word Density'] / 100, 3)
df['Compound'] = round(df['Compound'] * df['Topic Word Density'] / 100, 3)

#Normalization of ratings
df['Positive'] = abs(((df['Positive'] - df['Positive'].mean()) / (df['Positive'].max() - df['Positive'].min())) * 10)
df['Negative'] = abs(((df['Negative'] - df['Negative'].mean()) / (df['Negative'].max() - df['Negative'].min())) * 10)
df['Neutral'] = abs(((df['Neutral'] - df['Neutral'].mean()) / (df['Neutral'].max() - df['Neutral'].min())) * 10)
df['Compound'] = abs(((df['Compound'] - df['Compound'].mean()) / (df['Compound'].max() - df['Compound'].min())) * 10)

print(df)

#Output to csv
df.to_csv(str("\\files\opinion_mining_output.csv"), sep=",", header=True, index=True, line_terminator="\n")
