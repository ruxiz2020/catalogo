# catálogo

## prepare virtualenv

```bash
virtualenv -p python3 env
source env/bin/activate
```

Install the requirements:

```bash
env/bin/pip install -r requirements.txt
```

Bring up the UI:
```bash
env/bin/python index.py
```

Then open up `http://127.0.0.1:9000` in your browser


## references

* (Dash-cytoscape NLP demo for Plotly)[https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-cytoscape-lda]
* (LDA explained)[https://user.eng.umd.edu/~smiran/LDA.pdf]
* (tSNE vs. UMAP)[https://towardsdatascience.com/tsne-vs-umap-global-structure-4d8045acba17]
