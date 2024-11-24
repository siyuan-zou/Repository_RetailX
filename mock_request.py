import random
import asyncio


async def generate_bot_message(message, chat_history):
    await asyncio.sleep(0.5)
    return random.choice([
        "Sure! Here are some products I recommend:",
        "I found these products that might interest you:",
        "Take a look at these options:"
    ])


async def generate_recommendations(message, chat_history):
    """Generate a list of product recommendations with details."""
    await asyncio.sleep(0.5)  # Simulate async processing
    return [
        {
            "name": "Product A",
            "description": "你好.",
            "image": "https://via.placeholder.com/150",
            "link": "https://example.com/product-a",
            "discounted_price": "$20.00",
            "rating": 4.5
        },
        {
            "name": "Product B",
            "description": "早上好.",
            "image": "https://via.placeholder.com/150",
            "link": "https://example.com/product-b",
            "discounted_price": "$15.99",
            "rating": 4.0
        },
        {
            "name": "Product C",
            "description": "冰淇淋.",
            "image": "https://via.placeholder.com/150",
            "link": "https://example.com/product-c",
            "discounted_price": "$5.49",
            "rating": 3.8
        }
    ]


async def generate_maybeyoulike(message, chat_history):
    return random.choice([
        {
            "name": "Product X",
            "description": "Tailored to your needs.",
            "image": "https://via.placeholder.com/150",
            "link": "https://example.com/product-x",
            "discounted_price": "$35.00",
            "rating": 4.7
        },
        {
            "name": "Product Y",
            "description": "High quality and affordable.",
            "image": "https://via.placeholder.com/150",
            "link": "https://example.com/product-y",
            "discounted_price": "$25.99",
            "rating": 4.2
        },
        {
            "name": "Product Z",
            "description": "Popular choice among users.",
            "image": "https://via.placeholder.com/150",
            "link": "https://example.com/product-z",
            "discounted_price": "$18.75",
            "rating": 4.6
        }
    ], [])


async def request_and_generate_data(message, chat_history):
    bot_message = await generate_bot_message(message, chat_history)
    recommendations = await generate_recommendations(message, chat_history)
    maybeyoulike = await generate_recommendations(message, chat_history)
    return bot_message, recommendations, maybeyoulike
