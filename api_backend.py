from functions import *
from env import *
import warnings
import pandas as pd
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# prepare the dataset search model
dataset = pd.read_csv(file_path)
dataset = process_data(dataset)

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def generate_recommendations(dataframe):
    """
    Generate a list of product recommendations with details for the fron end.
    """
    recommendations = []
    for _, row in dataframe.iterrows():
        print(row)
        recommendations.append({
            "name": row["product_name"],
            "description": row["about_product"],
            "image": row["img_link"],
            "link": row["product_link"],
            "discounted_price": row["discounted_price"],
            "original_price": row["original_price"],
            "discount_percentage": row["discount_percentage"],
            "rating": row["rating"],
            "rating_count": row["rating_count"]
        })

    return recommendations


def send_backend_response(message, chat_history):
    """
    The send_backend_response function is used to send the bot message and the recommendations to the frontend.
    """

    # extract the query from the chat history and embed it
    query = extract_query(chat_history)
    print('Query: ', query)
    print('-' * 80)
    query_embedding = model.encode([query])

    # search for the top 5 most relevant products
    filtered_dataset = hard_filter(query, query_embedding, dataset)
    search_result = query_search(query_embedding, filtered_dataset, returning='data')

    print(search_result["description"].head(5))

    # generate response to frontend
    bot_message = response(search_result["description"], chat_history)
    recommendations = generate_recommendations(search_result[:min(3, len(search_result))])
    cross_sells = generate_recommendations(cross_sells_recommendation(search_result[:min(len(search_result), 20)], dataset))

    return bot_message, recommendations, cross_sells


def main():
    """
    The main function to run the bot locally and interact with the user. Untill you type 'exit' the bot will keep running.
    """

    print('initializing...')
    warnings.filterwarnings("ignore")

    # get the user input
    user_input = input('Hi! How can I help you today?')

    # save the chat history locally
    with open('history.txt', 'r') as file:
        history = file.read()
    history = 'Hi! How can I help you today?' + '\n'
    with open('history.txt', 'w') as file:
        file.write(history)

    while user_input != 'exit':

        # save the chat history
        with open('history.txt', 'r') as file:
            history = file.read()
        history += user_input + '\n'

        # extract the query from the chat history and embed it
        query = extract_query(history)
        print('Query:', query)
        query_embedding = model.encode([query])

        # search for the top 5 most relevant products
        filtered_dataset = hard_filter(query, query_embedding, dataset)
        search_result = query_search(query_embedding, filtered_dataset, returning='data')

        print(search_result["description"].head(5))

        # generate response
        salesperson_response = response(search_result["description"], history)

        # save the chat history
        history += salesperson_response + '\n'
        with open('history.txt', 'w') as file:
            file.write(history)
        print("History updated!")

        print('I also recommend this object:')
        print(cross_sells_recommendation(search_result[:min(len(search_result), 20)], dataset))
        print("recommendation done")

        # get the user input
        user_input = input(salesperson_response)

if __name__ == '__main__':
    main()
