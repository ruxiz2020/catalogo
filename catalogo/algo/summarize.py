import pytextrank
import spacy


class Summarize:

    def __init__(self, text, limit_phrases, limit_sentences):

        self._text = text
        self._limit_phrases = limit_phrases
        self._limit_sentences = limit_sentences

    def getSummary(self):

        # load a spaCy model, depending on language, scale, etc.
        nlp = spacy.load("en_core_web_sm")

        # add PyTextRank into the spaCy pipeline
        tr = pytextrank.TextRank(logger=None)
        nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
        doc = nlp(self._text)

        list_outcome = []
        for phrase in doc._.textrank.summary(
                limit_phrases=self._limit_phrases,
                limit_sentences=self._limit_sentences):

            list_outcome.append(phrase)
        #return ' \n'.join(list_outcome)
        list_of_strings  = [i.text for i in list_outcome]
        return list_of_strings

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

    getSumm = Summarize(text, 15, 10)
    summ = getSumm.getSummary()
    print('=== The summary is : ')
    print(' '.join(summ))
    #print('\n'.join('{}: {}'.format(k['text'], k['rank']) for k in enumerate(keywords)))
