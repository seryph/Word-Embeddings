# Word Embeddings Tutorial: Understanding Word Representations in AI

A comprehensive introduction to **Word Embeddings in Natural Language Processing (NLP)** using Python.

This tutorial explores how computers convert words into numerical representations, how neural networks learn relationships between words, and how word embeddings enable machines to process and understand human language.

## 1. Introduction

Word embeddings are numerical vector representations of words that capture semantic and syntactic relationships.

Unlike traditional text representation techniques, word embeddings allow similar words to have similar numerical representations.

For example, words such as *king*, *queen*, *man*, and *woman* can be represented as vectors that capture meaningful relationships.

Word embeddings are fundamental to many modern NLP applications, including:

- Machine translation
- Sentiment analysis
- Text classification
- Recommendation systems
- Search engines
- Large Language Models (LLMs)

## 2. What You'll Learn

This tutorial covers the following topics:

1. Introduction to word embeddings
2. Bag of Words (BoW)
3. One-Hot Encoding
4. Word2Vec
5. Continuous Bag of Words (CBOW)
6. Skip-Gram
7. Cosine Similarity
8. Visualizing word embeddings
9. Practical applications of word embeddings

## 3. Traditional Text Representation

Before exploring neural word embeddings, we examine traditional techniques for converting text into numerical representations.

### Bag of Words

Bag of Words represents text using word frequencies while ignoring word order.

Consider these sentences:

- "The cat sat on the mat."
- "The dog sat on the rug."

A Bag of Words model constructs a vocabulary and represents each sentence as a vector containing word counts.

Although simple and effective, Bag of Words does not capture semantic relationships between words.

### One-Hot Encoding

One-hot encoding represents each word using a vector containing a single `1` and zeros everywhere else.

For example:

```python
cat = [1, 0, 0]
dog = [0, 1, 0]
bird = [0, 0, 1]
```

The main limitation is that every word is represented independently, meaning the vectors cannot capture semantic similarity.

## 4. Understanding Word2Vec

Word2Vec is a neural network-based technique for learning dense vector representations of words.

It learns word embeddings by analyzing the contexts in which words appear.

There are two primary Word2Vec architectures.

### Continuous Bag of Words (CBOW)

CBOW predicts a target word using its surrounding context words.

Example:

Input: "The cat ___ on the mat."

Target: "sat"

The model learns to predict the missing word using information from the surrounding words.

### Skip-Gram

Skip-Gram performs the opposite task.

Given a target word, the model predicts surrounding context words.

Example:

Input: "sat"

Possible context words: "cat", "on"

Both architectures learn useful word embeddings through their prediction tasks.

## 5. Implementing Word2Vec in Python

We use Gensim to train a simple Word2Vec model.

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat", "on", "the", "rug"],
    ["the", "cat", "and", "dog", "are", "animals"],
    ["animals", "can", "sit", "on", "a", "mat"]
]

model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=2,
    min_count=1,
    sg=1,
    epochs=100
)

# Retrieve the embedding for a word
print(model.wv["cat"])
```

Setting `sg=1` trains a Skip-Gram model, while `sg=0` trains a CBOW model.

Note that this tiny dataset is intended for demonstrating the API. Learning meaningful semantic relationships requires substantially more training data.

## 6. Measuring Similarity With Cosine Similarity

Cosine similarity measures the similarity between two vectors based on the cosine of the angle between them.

The formula is:

\[
\text{Cosine Similarity}(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
\]

Where:

- A and B represent word vectors.
- The numerator represents their dot product.
- The denominator represents the product of their magnitudes.

For nonzero vectors, cosine similarity ranges from -1 to 1.

A value close to 1 indicates that two vectors point in similar directions.

Example:

```python
similarity = model.wv.similarity(
    "cat",
    "dog"
)

print(similarity)
```

You can also find words with similar vector representations:

```python
similar_words = model.wv.most_similar(
    "cat",
    topn=5
)

print(similar_words)
```

## 7. Visualizing Word Embeddings

Word embeddings typically have many dimensions, making them difficult to visualize directly.

Dimensionality reduction techniques such as PCA and t-SNE allow us to project embeddings into two-dimensional space.

Example using PCA:

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

words = list(model.wv.index_to_key)
vectors = model.wv[words]

pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

plt.figure(figsize=(10, 7))

plt.scatter(
    reduced_vectors[:, 0],
    reduced_vectors[:, 1]
)

for i, word in enumerate(words):
    plt.annotate(
        word,
        (reduced_vectors[i, 0],
         reduced_vectors[i, 1])
    )

plt.title("Word Embeddings Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
```

When trained on sufficiently large and representative datasets, word embeddings can capture meaningful semantic relationships.

## 8. Requirements

This tutorial uses Python 3 and the following libraries:

- NumPy
- Matplotlib
- Scikit-learn
- Gensim
- Jupyter Notebook

Install the required dependencies:

```bash
pip install numpy matplotlib scikit-learn gensim notebook
```

## 9. Running the Tutorial

Clone this repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Navigate into the project directory:

```bash
cd YOUR_REPOSITORY
```

Install the required dependencies and launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the tutorial notebook and execute the cells sequentially.

Replace the placeholder repository URL and notebook instructions with your actual project details before publishing.

## 10. Key Takeaways

By completing this tutorial, you should understand:

- How computers convert text into numerical representations.
- The limitations of traditional text encoding methods.
- How Word2Vec learns dense vector representations.
- The differences between CBOW and Skip-Gram.
- How cosine similarity measures relationships between word vectors.
- How dimensionality reduction helps visualize embeddings.

These concepts provide a foundation for understanding more advanced NLP architectures, including transformers and modern language models.

## 11. Additional Resources

- [Original Word2Vec Paper](https://arxiv.org/abs/1301.3781)
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## 12. Contributing

Contributions, suggestions, and improvements are welcome!

Feel free to open an issue or submit a pull request if you discover
