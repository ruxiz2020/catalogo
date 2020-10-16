import pandas as pd
import logging
import sys
import itertools
sys.path.append("..")

# ===== START LOGGER =====
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
root_logger.addHandler(sh)

df_input = pd.read_csv('../outputs/keywords_df.csv.gz', compression='gzip')
df_keywords = pd.DataFrame(df_input.groupby(
    ['title', 'author', 'date'])['keywords'].apply(list))
df_keywords.reset_index(inplace=True)
list_keywords = df_keywords['keywords'].values.tolist()
print(list_keywords[:2])

list_combination = []
list_combination_index = []
for comb in itertools.combinations(list_keywords, 2):
    list_combination.append(comb)
    list_combination_index.append(
        [list_keywords.index(comb[0]), list_keywords.index(comb[1])])
print('=== There are in total : {} pair of size 2 combinations'.format(
    len(list_combination)))

df_comb = pd.DataFrame(list_combination, columns=[
                       'keywords_01', 'keywords_02'])
# print(df_comb.head())

df_comb_index = pd.DataFrame(list_combination_index, columns=[
                             'index_01', 'index_02'])
df_comb['index_01'] = df_comb_index['index_01']
df_comb['index_02'] = df_comb_index['index_02']

# print(df_comb_index.tail())


def jaccard_sim(lst1, lst2):
    a = set(lst1)
    b = set(lst2)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


df_comb['jac_sim'] = df_comb[['keywords_01', 'keywords_02']].apply(
    lambda x: jaccard_sim(*x), axis=1)

print('=== The distribution of jaccard similarity of the keywords pairs:')
print(df_comb.jac_sim.describe())

df_comb2 = df_comb[df_comb['jac_sim'] > 0.03]
df_comb2 = df_comb2.merge(df_keywords[[
                          'title', 'author', 'date']], left_on='index_01', right_index=True, how='left')

df_dummy = df_keywords[['title']].copy()
df_dummy = df_dummy.rename(columns={'title': 'target_title'})
df_comb2 = df_comb2.merge(df_dummy, left_on='index_02',
                          right_index=True, how='left')

# print(df_comb2.columns)
def list_2_str(lst):

    return '/'.join(lst)

def len_list(lst):

    return len(lst)

df_agg = pd.DataFrame(df_comb2.groupby(['title', 'author', 'date'])[
                      'target_title'].apply(list))
df_agg.reset_index(inplace=True)
df_agg['n_target_title'] = df_agg['target_title'].apply(len_list)
df_agg['target_title'] = df_agg['target_title'].apply(list_2_str)


df_lda = pd.read_csv('../outputs/lda_df.csv')
print(df_lda.head())
df_lda = df_lda.merge(df_agg, on='title', how='left')

jacc_file = "../outputs/network_df.csv"
logger.info(f"Saving file compressed {jacc_file}")
df_lda.to_csv(jacc_file, index=False)
