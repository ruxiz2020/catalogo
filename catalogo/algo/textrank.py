import pytextrank
import spacy
import pandas as pd

class TextRank:

    def __init__(self, text, rank_threshold=0.01):

        self._text = text

    def getKeywords(self):

        # load a spaCy model, depending on language, scale, etc.
        nlp = spacy.load("en_core_web_sm")

        # add PyTextRank into the spaCy pipeline
        tr = pytextrank.TextRank(logger=None)
        nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
        doc = nlp(self._text)

        list_outcome = []
        for phrase in doc._.phrases:
            #print("{:.4f} {:5d}  {}".format(phrase.rank, phrase.count, phrase.text))
            # print(phrase.chunks)
            dict_result = {
                #'chuncks': phrase.chunks,
                'rank': phrase.rank,
                'count': phrase.count,
                'keywords': phrase.text
            }
            list_outcome.append(dict_result)
        df_outcome = pd.DataFrame(list_outcome)
        try:
            df_outcome = df_outcome[df_outcome['rank']>=rank_threshold]
        except:
            df_outcome = df_outcome
        return df_outcome


if __name__ == '__main__':

    import sys
    sys.path.append("..")
    from util.extract_text import ExtractText


    url = "https://www.bbc.com/news/election-us-2020-54359993"
    print('=== Reading from : {}'.format(url))

    extractText = ExtractText(url)
    title, text = extractText.extract_text_from_html()

    print('=== The title is : {}'.format(title))
    #print('=== The 1st 100 chars are: {}'.format(text[:100]))

    getRank = TextRank(text, rank_threshold=0.02)
    keywords = getRank.getKeywords()
    print('=== The top 5 keywords are : ')
    print(keywords[:5])
    #print('\n'.join('{}: {}'.format(k['text'], k['rank']) for k in enumerate(keywords)))
