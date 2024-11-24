import re
from api_backend import send_backend_response as get_backend_response


def clear_chat_history(chat_history):
    """
    Restart the communication history for new user/recommendations
    """
    cleared_chat_history = ''
    for chat in chat_history:
        content = re.sub(r"<ul class='chat-recommendations'>.*?</ul>",
                         "", chat['content'], flags=re.DOTALL)
        cleared_chat_history += (chat['role'] + ':' + content + '\n')
    cleared_chat_history = re.sub(r'\n\s*\n', '\n', cleared_chat_history)
    return cleared_chat_history


async def frontend_request(message, chat_history):
    """
    The frontend_request function is used to send the user message and chat history to the backend and get the response
    """
    print('MESSAGE: ', message)
    print('CHAT HISTORY:\n', chat_history)
    # Extract query from chat history
    chat_history = clear_chat_history(chat_history)
    chat_history += 'user:' + message
    print('-' * 80)
    print('CLEARED CHAT HISTORY: ', chat_history)
    print('-' * 80)
    bot_message, recommendations, maybeyoulike = get_backend_response(
        message, chat_history)
    print('BOT MESSAGE:', bot_message)
    print('RECOMMENDATIONS', [rec['name'] for rec in recommendations])
    return bot_message, recommendations, maybeyoulike
