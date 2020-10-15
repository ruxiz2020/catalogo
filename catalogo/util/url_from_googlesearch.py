from googlesearch import search


class ExtractUrl:

    def __init__(self, topic, site, n_url = 20):
        """
        Extract text data from url or local path to html file
        """
        self._search_param = '\"' + topic + '\"' + " " + site
        self._n = n_url

    def _get_url(self):

        list_url = []
        for url in search(self._search_param, stop=self._n):
            list_url.append(url)

        return list_url


if __name__ == '__main__':

    topic = "explain model"
    site = "medium.com"

    eu = ExtractUrl(topic, site, 10)
    print('search ' + topic + ' on ' + site)
    res = eu._get_url()
    print(res)

    #with open('urls_medium.txt', 'a') as f:
    #    for item in res:
    #        if item.startswith('https://medium.com') | item.startswith('https://towardsdatascience.com'):
    #            f.write(item + "\n")
