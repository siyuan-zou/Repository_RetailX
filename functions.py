from env import *
from mistralai import Mistral
import pandas as pd
import os

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Initialize the Mistral API
client = Mistral(api_key=api_key)
model_s = "mistral-small-latest"
model_l = "mistral-large-latest"

# Load all text files
with open(pricer_path, 'r') as f:
    personality_pricer = f.read()
with open(discounter_path, 'r') as f:
    personality_discounter = f.read()
with open(rater_path, 'r') as f:
    personality_rater = f.read()
with open(salesperson_path, 'r') as f:
    salesperson = f.read()
with open(keyword_extractor_path, 'r') as f:
    keyword_extractor = f.read()

if not os.path.exists(history_file_path):
    history = ''
    print(
        f"Warning: {history_file_path} not found. 'history' will be set to an empty string.")
    with open(history_file_path, 'a') as f:
        f.write('\n')

with open(history_file_path, 'r') as f:
    history = f.read()


def process_data(dataset):
    """
    dataset: The dataframe to process

    output: The processed dataframe
    """
    
    product = dataset.copy()

    # Convert the price, discount, rating and actual price to numeric
    product.loc[:, 'price'] = pd.to_numeric(
        product['discounted_price'].replace('[₹,]', '', regex=True), errors='coerce')
    product.loc[:, 'actual_price'] = pd.to_numeric(
        product['actual_price'].replace('[₹,]', '', regex=True), errors='coerce')
    product.loc[:, 'discount_percentage'] = pd.to_numeric(
        product['discount_percentage'].replace('%', '', regex=True), errors='coerce')
    product.loc[:, 'rating'] = pd.to_numeric(
        product['rating'], errors='coerce')
    product.rename(columns={'actual_price': 'original_price'}, inplace=True)

    # Combine the information from the product name, about product, review title and review content
    product.loc[:, 'description'] = product['product_name'] + \
        " " + product['about_product']
    product.loc[:, 'comments'] = product['review_title'] + \
        " " + product['review_content']
    product.loc[:, 'description_and_comments'] = product['description'] + \
        " "+product['comments']

    return product

def extract_query(chat_history):
    """
    chat_history: The chat history

    output: The user query
    """
    # Give the conversation context to the keyword extractor
    message = "Conversation to extract:\n"+str(chat_history)

    return single_chat(message, keyword_extractor, model_l)


def query_search(query_embedding, dataset, topN='all', returning='index'):
    """
    query_embedding: The embedding of the query
    product_embeddings: The embeddings of the product to search
    topN: The number of top product to return
    returning: The type of output to return

    output: The type of output to return. It can be either 'index' or 'dataframe
    """
    # Set the number of top products to return
    if topN == 'all':
        topN = len(dataset)

    # Load the product description and comments embeddings
    product_embeddings = torch.load('embeddings/product_embeddings.pt')

    # Calculate cosine similarity between the query and each product's combined text
    B = [(product_embeddings[i]) for i in dataset.index]
    # Calculate cosine similarity between the query and each product's combined text
    similarities = cosine_similarity(query_embedding.reshape(1, -1), B)
    dataset['relevance_score'] = similarities.flatten()

    # Sort by similarity and return the top 5 most relevant product
    matched_product = dataset.sort_values(
        by='relevance_score', ascending=False)
    if returning == 'index':
        return matched_product.head(topN).index.tolist()
    else:
        return matched_product.head(topN)


def single_chat(input_text, personality, model_LLM=model_s):
    """
    input_text: The user query
    personality: The personality to use
    model_LLM: The language model to use

    output: The response from the language model
    """
    response = client.chat.complete(
        model=model_LLM,
        messages=[{"role": "user", "content": input_text},
                  {"role": "system", "content": personality}],
    ).choices[0].message.content
    return response


def hard_filter_trigger(query_embedding):
    """
    query_embedding: The embedding of the user query
    output: boolean values indicating whether the query contains a filter trigger
    """

    price_triggers = torch.load('embeddings/price_embeddings.pt')
    discount_triggers = torch.load('embeddings/discount_embeddings.pt')
    rating_triggers = torch.load('embeddings/rating_embeddings.pt')

    triggers = [False]*3

    # Check price triggers
    for trigger_embedding in price_triggers:
        similarity = cosine_similarity(trigger_embedding.reshape(
            1, -1), query_embedding.reshape(1, -1))
        if similarity > 0.23:
            print(f"Price trigger activated with similarity: {similarity}")
            triggers[0] = True
            break

    # Check discount triggers
    for trigger_embedding in discount_triggers:
        similarity = cosine_similarity(trigger_embedding.reshape(
            1, -1), query_embedding.reshape(1, -1))
        if similarity > 0.3:
            print(f"Discount trigger activated with similarity: {similarity}")
            triggers[1] = True
            break

    # Check rating triggers
    for trigger_embedding in rating_triggers:
        similarity = cosine_similarity(trigger_embedding.reshape(
            1, -1), query_embedding.reshape(1, -1))
        if similarity > 0.3:
            print(f"Rating trigger activated with similarity: {similarity}")
            triggers[2] = True
            break

    print('Triggers:', triggers)

    return triggers


def hard_filter(input_text, query_embedding, dataset, model_LLM=model_l):
    """
    input_text: The user query
    query_embedding: The embedding of the user query
    dataset: The dataset to filter
    model_LLM: The language model to use

    output: The filtered dataset
    """
    # Check if the filter triggers are activated
    trigers = hard_filter_trigger(query_embedding)

    # Get the response from the pricer, discounter and rater
    if trigers[0]:
        pricer_response = single_chat(
            input_text, personality_pricer, model_LLM)
        if len(pricer_response) > 15:
            pricer_response = 'NaN'
        print('pricer: ', pricer_response)
    else:
        pricer_response = 'NaN'

    if trigers[1]:
        discounter_response = single_chat(
            input_text, personality_discounter, model_LLM)
        if len(discounter_response) > 15:
            discounter_response = 'NaN'
        print('discounter: ', discounter_response)
    else:
        discounter_response = 'NaN'

    if trigers[2]:
        rater_response = single_chat(input_text, personality_rater, model_LLM)
        if len(rater_response) > 15:
            rater_response = 'NaN'
        print("rater: ", rater_response)
    else:
        rater_response = 'NaN'

    # Filter the dataset based on the responses
    filtered_dataset = dataset.copy()
    if pricer_response != 'NaN':
        filtered_dataset = filtered_dataset[filtered_dataset['price'] <= float(
            pricer_response.split(',')[1][:-1])]
        filtered_dataset = filtered_dataset[filtered_dataset['price'] >= float(
            pricer_response.split(',')[0][1:])]
    if discounter_response != 'NaN':
        filtered_dataset = filtered_dataset[filtered_dataset['discount_percentage'] <= float(
            discounter_response.split(',')[1][:-1])]
        filtered_dataset = filtered_dataset[filtered_dataset['discount_percentage'] >= float(
            discounter_response.split(',')[0][1:])]
    if rater_response != 'NaN':
        filtered_dataset = filtered_dataset[filtered_dataset['rating'] <= float(
            rater_response.split(',')[1][:-1])]
        filtered_dataset = filtered_dataset[filtered_dataset['rating'] >= float(
            rater_response.split(',')[0][1:])]

    return filtered_dataset


def response(dataset, theme):
    """
    dataset: The dataset to generate response from
    theme: The history summerization

    output: The response from the salesperson
    """
    # Get the first 5 items in the dataset
    infos = str(dataset[:min(5, len(dataset))].tolist())

    # Generate the message
    message = "### ITEM DESCRIPTIONS:\n"+infos + \
        "\n### CONVERSATION HISTORY:\n"+str(theme)
    
    # Get the response from the salesperson
    response = single_chat(message, salesperson, model_l)

    return response
