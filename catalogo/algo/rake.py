from rake_nltk import Rake

class RakeImpl:

    def __init__(self, text):
        self.text = text
        self.rake = Rake()

    def getKeywords(self):
        self.rake.extract_keywords_from_text(self.text)
        return self.rake.get_ranked_phrases()

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

    getkw = RakeImpl(text)
    kw = getkw.getKeywords()
    print('=== The top 10 keywords are : ')
    print(kw[:10])
    #print('\n'.join('{}: {}'.format(k['text'], k['rank']) for k in enumerate(keywords)))
