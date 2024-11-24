"""
This is the file to generate all embedding files. The embeddings are generated using the SentenceTransformer library. The embeddings are generated for the following.
"""

from functions import *
from env import *
import pandas as pd
from sentence_transformers import SentenceTransformer

import warnings
import torch
import os


def try_embed(objects, filename, model):
    """
    objects: The list of objects to embed
    filename: The name of the file to save the embeddings
    model: The SentenceTransformer model to use

    output: None
    """
    if not os.path.exists('embeddings/'+filename+'.pt'):

        product_embeddings = model.encode(objects)
        torch.save(product_embeddings, 'embeddings/'+filename+'.pt')


def main():
    """
    This is the main function to generate all the embeddings.
    """
    warnings.filterwarnings("ignore")

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    Price_triggers = ['price', 'cheap', 'expensive', 'value', 'budget', 'cost',
                      'affordable', 'inexpensive', 'reasonable', 'costly', 'economical', 'under', 'over', 'between', 'less', 'more', 'range', 'dollar', 'rupee']
    Discount_triggers = ['discount', 'sale', 'offer', 'deal', 'promotion',
                         'clearance', 'cut', 'reduction', 'savings', 'off', 'coupon']
    Rating_triggers = ['good', 'popular', 'rat', 'love', 'hate',
                       'suggest', 'like', 'recommend', 'satisf', 'feedback']
    
    # Generate embeddings for the triggers
    try_embed(Price_triggers, 'price_embeddings', model)
    try_embed(Discount_triggers, 'discount_embeddings', model)
    try_embed(Rating_triggers, 'rating_embeddings', model)

    dataset = pd.read_csv(file_path)
    dataset = process_data(dataset)

    # Generate embeddings for the product descriptions
    try_embed(dataset["description_and_comments"].tolist(),
              'product_embeddings', model)
    
    # Generate embeddings for the product categories
    categories = dataset['category'].apply(lambda x: " ".join(x.split('|')[-2:])).tolist()

    try_embed(categories,
               'category_embeddings', model)

if __name__ == "__main__":
    main()
