# MS Azure-ML SDK Study

Learning how to use Azure-ML SDK by creating classification models for the Poker Hand Dataset

https://archive.ics.uci.edu/ml/datasets/Poker+Hand

A quick demonstration on

- MS Azure-ML Services and
- Sphinx technology

In this repository we will:

- access a MS Azure Workspace and display its properties
- create an experiment using Scikit-Learn to build a dirty quick machine learning classification model
- submit this experiment to an Azure Compute instance (remote processing)
- create a second experiment for classification, but this time using MS Azure AutoML
- compare the models
- report everything using the amazing Sphinx package.


## The machine learning classification challenge

In this experiment, we will build a classification model for prediction.

Based on a Poker card distribution, we will be predicting the *Poker Hand* classification.

Each hand has five cards. We will register the Suit and the Rank of each card.

The Suits are:

- Ordinal (1-4)
- representing Hearts, Spades, Diamonds, Clubs

The Ranks are:

- Numerical (1-13)
- representing Ace, 2, 3, ... , Queen, King


Each hand is a record entry on the dataset.

The possible classes for a given hand:

- 0: Nothing in hand; not a recognized poker hand
- 1: One pair; one pair of equal ranks within five cards
- 2: Two pairs; two pairs of equal ranks within five cards
- 3: Three of a kind; three equal ranks within five cards
- 4: Straight; five cards, sequentially ranked with no gaps
- 5: Flush; five cards with the same suit
- 6: Full house; pair + different rank three of a kind
- 7: Four of a kind; four equal ranks within five cards
- 8: Straight flush; straight + flush
- 9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush


### Information about the dataset

- [Poker Hand Dataset](https://archive.ics.uci.edu/ml/datasets/Poker+Hand)
- The Poker Hand Dataset - from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.


## The Sphinx PPython Documentation Generator

From the Sphinx [website](https://www.sphinx-doc.org/en/master/index.html):

> *Sphinx is a tool that makes it easy to create intelligent and beautiful documentation, written by Georg Brandl and licensed under the BSD license.*
