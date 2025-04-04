## TransformerFromScratch
PyTorch Implementation of Transformer Deep Learning Model. You can understand how it works via [Transformers Architecture Presentation](https://github.com/fualsan/TransformerFromScratch/blob/main/Transformers-Architecture-Presentation.pptx)

In Implementation part, [Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset) is used.

## Code Workflow
- Classes
	- Encoder Layer
		- Multihead Attention
		- FeedForward Netword
		- Residual Connections
	- MaskedOnlyDecoder Layer
		- Multihead Attention
		- FeedForward Netword
		- Residual Connections
	- Decoder Layer
		- Multihead Attention
		- FeedForward Netword
	- Residual Connections

- Models
	- BERT
		- Encoder Layer
	- MaskedOnlyDecoder
		- MaskedOnlyDecoder Layer
	- GPT
		- Decoder Layer
	- T5
		- Encoder Layer
		- Decoder Layer

## Resources
- [Attention is all you need, Original Paper](https://arxiv.org/abs/1706.03762b)

- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

- [Attention Mechanism step-by-step Video](https://www.youtube.com/watch?v=eMlx5fFNoYc&t=172s&ab_channel=3Blue1Brown)

## Acknowledgement
This repo is created w/ the help of [H. Fuat ALSAN's Github Repo](https://github.com/fualsan/TransformerFromScratch).

## If this repo is useful to you, please cite it and forgot to check H. Fuat ALSAN's Repo:

```
@misc{transformers_from_scratch,
	title = {batulaiko/TransformerFromScratch},
	url = {https://github.com/batulaiko/TransformerFromScratch},
	author = {YILDIRIM, Batuhan},
    year={2025},
}
```
