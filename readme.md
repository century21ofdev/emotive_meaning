### Installation


```commandline
python3 -m venv venv
```

```commandline
source venv/bin/activate
```

```commandline
pip install -r requirements
```

```commandline
python
```
```python
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('english')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

```commandline
python pipeline.py
```