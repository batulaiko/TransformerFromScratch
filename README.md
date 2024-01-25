# TransformerFromScratch
PyTorch Implementation of Transformer Deep Learning Model

# Preprocess a Wikipedia Dump File
* Edit the **wiki_dump_read.py** file to use your own dump of Wikipedia.
* Adjust the *hyperparameters* of preprocessing.
* Run the preprocessor:
```python
$ python wiki_dump_read.py
```

# Train a tokenizer
Use notebook **TokenizerTrainBPE.ipynb** to train your own byte-pair encoding (BPE) tokenizer. 

### Playground for tokenizers
Use notebook **PlaygroundForTokenizers.ipynb** to test your trained BPE tokenizer. 

# Cache Wiki Dump Dataset (Optional)
* Dataset caching is useful for improving context and training speed.
* Run the caching (this can take some time):
```python
$ python cache_dataset.py
```

# Train Languge Model (GPT style)
Use notebook **LM_GPT_Wiki.ipynb** to train your own transformer based language model. 

# Generate text
* Edit the **generate_text.py** file for your configuration.
* Use *setup* and *temps* to adjust text generation behaviour.
* Use *q* to quit, *clear* to clear console.
  
```python
$ python generate_text.py
```

# If this repo is useful to you, please cite it:
```
@misc{alsan_transformers_from_scratch,
	title = {fualsan/TransformerFromScratch},
	url = {https://github.com/fualsan/TransformerFromScratch},
	author = {Alsan, H. Fuat},
    year={2024},
}
```
