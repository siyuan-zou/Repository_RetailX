import gradio as gr
# from api_frontend import frontend_request
from mock_request import request_and_generate_data as frontend_request

# Styling
with open("style.css", "r") as file:
    CSS = file.read()

THEME = gr.themes.Soft(primary_hue=gr.themes.colors.orange,
                       secondary_hue=gr.themes.colors.yellow)

# Components

showcase_header = "<div id='showcase-heading'> ❤️ Here are some of our top picks for you</div>"
initial_chat_history = [
    {"role": "assistant",
        "content": "Hi there! 👋 Welcome to RetailX. How can I assist you today?"}
]


def cut_text(text, max_length=100):
    return text[:max_length] + ("..." if len(text) >= max_length else "")


def update_recommendations_html(recommendations, current_recommendations):
    """Convert a list of recommendations into HTML for display."""
    new_recommendations_html = "".join(
        f"""
    <div class="showcase">
        <div>
            <a href="{item['link']}" target="_blank" class="showcase-title">
            <div style="display: flex; align-items: center;">
                <img src="{item['image']}" 
                    alt="img"
                    class="showcase-img"
                    onerror="this.onerror=null; this.src='https://picsum.photos/150';">
                <strong style="margin-left: 10px;">{cut_text(item['name'], max_length=100)}</strong>
            </div>
            <p class="showcase-desc">{cut_text(item['description'], max_length = 100)}</p>
            <p>
            <span class="showcase-price">Price: <strong>{item['discounted_price']}</span>
            <span class="showcase-rating">Rating: ⭐ {item['rating']}</span>
            </p>
            </a>
        </div>
    </div>
    """
        for item in recommendations
    )

    return showcase_header + \
        new_recommendations_html + \
        current_recommendations.replace(showcase_header, "")


def update_chat_recommendations(recommendations, maybeyoulike):
    """Convert recommendations and 'Maybe you like' into clickable unordered lists for Chatbot."""
    # Check if 'maybeyoulike' is not empty or contains only whitespace
    recommendations_content = (
        "<ul class='chat-recommendations'>"
        + "".join(
            f"<li><a href='{item['link']}' target='_blank' class='chat-recommendation'>{cut_text(item['name'], max_length=40)}</a></li>"
            for item in recommendations
        )
        + "</ul>"
    )
    maybeyoulike_content = ""
    if len(maybeyoulike) != 0:
        maybeyoulike_content = (
            "<div class='maybeyoulike-header'>Maybe you like:</div>"
            + "<ul class='chat-recommendations'>"
            + "".join(
                f"<li><a href='{item['link']}' target='_blank' class='chat-recommendation'>{cut_text(item['name'], max_length=40)}</a></li>"
                for item in maybeyoulike
            )
            + "</ul>"
        )
    return recommendations_content + maybeyoulike_content


# Interaction Callbacks


async def chat_callback(message, chat_history, current_recommendations):
    """Handle user input, generate bot response, and product recommendations."""
    if not message.strip():
        gr.Info("Empty message. Please type something.")
        return "", chat_history, current_recommendations

    # Generate bot response
    bot_message, new_recommendations, maybeyoulike = await frontend_request(message, chat_history)

    # Append messages to chat history
    updated_recommendations = update_recommendations_html(
        new_recommendations, current_recommendations)
    chat_recommendations = update_chat_recommendations(
        new_recommendations, maybeyoulike)
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
