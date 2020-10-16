import sys
sys.path.append("..")
from catalogo.algo.textrank import TextRank
import pandas as pd
import logging

# ===== START LOGGER =====
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
root_logger.addHandler(sh)

df_input_medium = pd.read_csv('../inputs/medium_texts.csv')
#df_input_confu = pd.read_csv('../inputs/confu_texts.csv')
df_input_medium['src'] = 'medium.com'
#df_input_confu['src'] = 'confluence'
#df_input = pd.concat([df_input_medium, df_input_confu])
df_input = df_input_medium
print(len(df_input))
df_input = df_input[~df_input['text'].isnull()]
print(len(df_input))


list_keywords = []
count = 0
for row in df_input.iterrows():
    print(count)
    title_text = row[1]['title']
    raw_text = row[1]['text']

    getRank = TextRank(raw_text, rank_threshold=0.04)
    df_keywords = getRank.getKeywords() # a dataframe with columns, rank/cound/keywords
    #print('=== The top 2 keywords are : ')
    # print(keywords[:2])
    df_keywords['title'] = title_text
    df_keywords['author'] = row[1]['author']
    df_keywords['date'] = row[1]['date']
    list_keywords.append(df_keywords)
    count += 1

df_res = pd.concat(list_keywords)
print(df_res)


keywords_file = "../outputs/keywords_df.csv.gz"
logger.info(f"Saving file compressed {keywords_file}")
df_res[['title', 'author', 'date', 'rank', 'keywords']].to_csv(
    keywords_file, compression='gzip')
