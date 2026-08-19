# Deep Learning Projects

6 projects covering neural networks, CNNs, RNNs, Transformers, GANs, autoencoders, and training pipelines.

## Projects

| Project | Notebooks | Description |
|---------|-----------|-------------|
| [autoencoders](./autoencoders/) | 5 | Variational autoencoders (CNN, DNN), denoising autoencoders (DCNN, DNN), basic autoencoder |
| [gans](./gans/) | 5 | GAN tutorials, DCGAN, Conditional GANs (CGAN), Keras implementations |
| [nlp-preprocessing](./nlp-preprocessing/) | 1 | NLP preprocessing techniques for deep learning |
| [seq2seq-translation](./seq2seq-translation/) | 1 | English to Hindi LSTM machine translation |
| [transfer-learning-cnn](./transfer-learning-cnn/) | 0 | Transfer learning with CNN (folder exists, notebooks to be added) |
| [word-embedding-rnn-lstm](./word-embedding-rnn-lstm/) | 0 | Word embeddings, RNN, LSTM, GRU, Bidirectional LSTM (folder exists, notebooks to be added) |

## Topics Covered

- **Autoencoders**: Variational (CNN/DNN), Denoising (DCNN/DNN), Basic implementations
- **Generative Adversarial Networks**: Vanilla GAN, DCGAN, Conditional GAN, Keras custom training loops
- **Sequence Models**: Seq2Seq translation with LSTM
- **NLP for DL**: Text preprocessing pipelines
- **Transfer Learning**: CNN fine-tuning (pending)
- **Word Embeddings & RNNs**: Embeddings, Vanilla RNN, LSTM, GRU, Bidirectional variants (pending)

## Getting Started

```bash
# Navigate to a project
cd autoencoders

# Open in Jupyter (auto-syncs with .py)
jupyter lab main.ipynb

# Or edit .py in your IDE
code main.py
```

All notebooks use Jupytext paired format (`.py` + `.ipynb`).

## Learning Path Suggestion

1. Start with `autoencoders/` - fundamentals of representation learning
2. Move to `gans/` - generative modeling
3. Explore `seq2seq-translation/` - sequence-to-sequence models
4. Review `nlp-preprocessing/` - text preprocessing for DL
5. Add notebooks to `transfer-learning-cnn/` and `word-embedding-rnn-lstm/` for CNN transfer learning and RNN/LSTM fundamentals