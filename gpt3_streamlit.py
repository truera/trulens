import dotenv
import openai
import streamlit as st

import tru
import tru_feedback

config = dotenv.dotenv_values(".env")

# Set OpenAI API key
openai.api_key = config['OPENAI_API_KEY']

# Set up GPT-3 model
model_engine = "gpt-3.5-turbo"
prompt = "Please help me with "


# Define function to generate GPT-3 response
@st.cache_data
def generate_response(prompt, model_engine):
    return openai.ChatCompletion.create(
        model=model_engine,
        messages=[
            {
                "role":
                    "system",
                "content":
                    "You are a helpful assistant that provides concise and relevant background information and context so that outsiders can easily understand."
            }, {
                "role": "user",
                "content": prompt
            }
        ]
    )["choices"][0]["message"]["content"]


# Set up Streamlit app
st.title("Get Help from ChatGPT")
user_input = st.text_input("What do you need help with?")

if user_input:
    # Generate GPT-3 response
    prompt_input = user_input
    gpt3_response = generate_response(prompt_input, model_engine)

    # Display response
    st.write("Here's some help for you:")
    st.write(gpt3_response)

    # Allow user to rate the response with emojis
    col1, col2 = st.columns(2)
    with col1:
        thumbs_up = st.button("👍")
    with col2:
        thumbs_down = st.button("👎")

    if thumbs_up:
        # Save rating to database or file
        st.write("Thank you for your feedback! We're glad we could help.")
    elif thumbs_down:
        # Save rating to database or file
        st.write(
            "We're sorry we couldn't be more helpful. Please try again with a different question."
        )

    record_id = tru.add_data(
        'chat_model', prompt_input, 'None', gpt3_response, '', {
            'thumbs_up': thumbs_up,
            'thumbs_down': thumbs_down
        }
    )

    # Run feedback function and get value
    feedback = tru.run_feedback_function(
        prompt_input, gpt3_response, [
            tru_feedback.FEEDBACK_FUNCTIONS['hate'](
                evaluation_choice='response',
                provider='openai',
                model_engine='moderation'
            )
        ]
    )

    # Add value to database
    tru.add_feedback(record_id, feedback)
