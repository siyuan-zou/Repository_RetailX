import gradio as gr
from api_frontend import frontend_request

# Styling
with open("style.css", "r") as file:
    CSS = file.read()

THEME = gr.themes.Soft(primary_hue=gr.themes.colors.indigo,
                       secondary_hue=gr.themes.colors.purple)

# Components

showcase_header = "<div id='showcase-heading'> ❤️ Here are some of our top picks for you</div>"
initial_chat_history = [
    {"role": "assistant",
        "content": "Hi there! 👋 Welcome to RetailX. How can I assist you today?"}
]


def update_recommendations_html(recommendations, current_recommendations):
    """
    Convert a list of recommendations into HTML for display.
    """
    new_recommendations_html = "".join(
        f"""
        <div class="showcase">
            <img src="{item['image']}" alt="{item['name']}" class="showcase-img">
            <div>
                <a href="{item['link']}" target="_blank" class="showcase-title">
                    <strong>{item['name']}</strong>
                </a>
                <p class="showcase-desc">{item['description'][:100] + ("..." if len(item['description'])>=100 else "") }</p>
            </div>
        </div>"""
        for item in recommendations
    )
    return showcase_header + \
        new_recommendations_html + \
        current_recommendations.replace(showcase_header, "")


def update_chat_recommendations(recommendations):
    """
    Convert recommendations into a clickable unordered list for Chatbot.
    """
    return (
        "<ul class='chat-recommendations'>"
        + "".join(
            f"<li><a href='{item['link']}' target='_blank' class='chat-recommendation'>{item['name']}</a></li>"
            for item in recommendations
        )
        + "</ul>"
    )

# Interaction Callbacks


async def chat_callback(message, chat_history, current_recommendations):
    """
    Handle user input, generate bot response, and product recommendations.
    """
    if not message.strip():
        gr.Info("Empty message. Please type something.")
        return "", chat_history, current_recommendations

    # Generate bot response
    bot_message, new_recommendations = await frontend_request(message, chat_history)

    # Append messages to chat history
    updated_recommendations = update_recommendations_html(
        new_recommendations, current_recommendations)
    chat_recommendations = update_chat_recommendations(new_recommendations)
    chat_history.append({"role": "user", "content": message})
    chat_history.append(
        {"role": "assistant", "content": f"{bot_message}\n\n{chat_recommendations}"})

    return "", chat_history, updated_recommendations


# Main layout

with gr.Blocks(css=CSS, theme=THEME, fill_height=True, title="RetailX - Your Personal Shopping Assistant") as demo:
    with gr.Row():
        gr.HTML(
            """
            <div id="my-title">
                🛍️ <strong>RetailX - Your Personal Shopping Assistant</strong>
            </div>
            """
        )
    with gr.Row():
        gr.HTML(
            """<div id="description">Welcome! Share your shopping needs, and our assistant will recommend the best products for you.</div>""")

    with gr.Row(elem_id="main-content"):
        with gr.Column(scale=2):
            # Initialize chatbot with greeting message
            chatbot = gr.Chatbot(label="Chat with your Assistant",
                                 elem_id="chatbot",
                                 type="messages",
                                 value=initial_chat_history)

        with gr.Column(scale=1):
            recommendations_box = gr.HTML(
                showcase_header, elem_id="recommendations"
            )

    msg = gr.Textbox(
        placeholder="Type your message here...",
        label="✉️ Your Message",
        elem_id="footer",
        submit_btn=True
    )

    msg.submit(chat_callback, [msg, chatbot, recommendations_box], [
               msg, chatbot, recommendations_box])

# Launch the app
demo.launch()
