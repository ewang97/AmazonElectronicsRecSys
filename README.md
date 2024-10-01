# Amazon Electronics Recommendation System
A collaborative-filtering based recommendation system using surprise to offer top-K recommendations based on existing user-rating data for Amazon electronics. Recommendations are determined by predicting the the highest ratings for each item across users and then selecting the items with the K highest values.


## Installation

Packages to install included in requirements.txt:

```bash
  pip install requirements.txt
```
    

# Dataset:
https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews

The Amazon dataset contains the following attributes:

- userId: Every user identified with a unique id
- productId: Every product identified with a unique id
- Rating: The rating of the corresponding product by the corresponding user
- timestamp: Time of the rating. We will not use this column to solve the current problem

Originally sourced:
https://jmcauley.ucsd.edu/data/amazon/


## Run Locally

Clone the project

```bash
  git clone https://github.com/ewang97/AmazonElectronicsRecSys.git
```

Go to the project directory

```bash
  cd AmazonElectronicsRecSys
```

Download dataset into project folder: https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews

Install dependencies

```bash
  pip install requirements.txt
```


## Usage/Examples

```python
python -m predict.py

```
