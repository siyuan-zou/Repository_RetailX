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
    await asyncio.sleep(0.5)
    """Generate a list of product recommendations with details."""
    return random.choice([
        [
            {"name": "Product A", "description": "你好.",
                "image": "https://via.placeholder.com/150", "link": "https://example.com/product-a"},
            {"name": "Product B", "description": "早上好.",
                "image": "https://via.placeholder.com/150", "link": "https://example.com/product-b"},
            {"name": "Product C", "description": "冰淇淋.",
                "image": "https://via.placeholder.com/150", "link": "https://example.com/product-c"}
        ],
        [
            {"name": "Product X", "description": "Tailored to your needs.",
                "image": "https://via.placeholder.com/150", "link": "https://example.com/product-x"},
            {"name": "Product Y", "description": "High quality and affordable.",
                "image": "https://via.placeholder.com/150", "link": "https://example.com/product-y"},
            {"name": "Product Z", "description": "Popular choice among users.",
                "image": "https://via.placeholder.com/150", "link": "https://example.com/product-z"}
        ]
    ])


async def request_and_generate_data(message, chat_history):
    bot_message = await generate_bot_message(message, chat_history)
    recommendations = await generate_recommendations(message, chat_history)
    return bot_message, recommendations
