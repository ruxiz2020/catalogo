import sys
sys.path.append("..")
from catalogo.algo.summarize import Summarize
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


list_summ = []
for row in df_input.iterrows():

    #title_text = row[1]['title']
    raw_text = row[1]['text']

    getSumm = Summarize(raw_text, 15, 3)
    summ = getSumm.getSummary()
    print('=== The summary is : ')
    print(' '.join(summ))
    list_summ.append(' '.join(summ))

df_input['summary'] = list_summ

summary_file = "../outputs/summary_df.csv.gz"
logger.info(f"Saving file compressed {summary_file}")
df_input[['title', 'author', 'date', 'summary']].to_csv(
    summary_file, compression='gzip')
