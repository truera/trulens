import os
from dotenv import load_dotenv
from snowflake.snowpark import Session
from snowflake.cortex import Summarize, Complete, ExtractAnswer, Sentiment, Translate

# Load environment variables from .env
load_dotenv()

connection_params = {
    "account": os.environ["SNOWFLAKE_ACCOUNT"],
    "user": os.environ["SNOWFLAKE_USER"],
    "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
}


# Create a Snowflake session
snowflake_session = Session.builder.configs(connection_params).create()


# Define the LLM functions
def summarize(user_text):
    summary = Summarize(text=user_text, session=snowflake_session)
    return summary


def complete(user_text):
    completion = Complete(
        model="snowflake-arctic",
        prompt=f"Provide 5 keywords from the following text: {user_text}",
        session=snowflake_session,
    )
    return completion


# def extract_answer(user_text):
#     answer = ExtractAnswer(
#         from_text=user_text,
#         question="What are some of the ethical concerns associated with the rapid development of AI?",
#         session=snowflake_session,
#     )
#     return answer


# def sentiment(user_text):
#     sentiment = Sentiment(text=user_text, session=snowflake_session)
#     return sentiment


# def translate(user_text):
#     translation = Translate(
#         text=user_text, from_language="en", to_language="de", session=snowflake_session
#     )
#     return translation


# Define the main function
def main():
    user_text = """
        The recent advancements in artificial intelligence have revolutionized various industries. From healthcare to finance, AI-powered solutions are enhancing efficiency and accuracy. In healthcare, AI is being used to predict patient outcomes, personalize treatment plans, and even assist in surgeries. Financial institutions are leveraging AI to detect fraudulent activities, provide personalized banking experiences, and improve risk management.

        However, the rapid development of AI also raises ethical concerns. Issues such as data privacy, algorithmic bias, and the potential for job displacement are being actively debated. Ensuring that AI technologies are developed and deployed responsibly is crucial for maximizing their benefits while minimizing their drawbacks.

        Furthermore, the global race for AI supremacy is intensifying. Countries and corporations are investing heavily in AI research and development to gain a competitive edge. This competition is driving innovation but also highlighting the need for international cooperation and regulation.

        In conclusion, while AI holds tremendous potential to transform our world positively, it is imperative to address the associated challenges. By fostering ethical AI practices and encouraging collaboration across borders, we can harness the full power of AI for the greater good.
    """

    try:
        # summary_result = summarize(user_text)
        # print(
        #     f"Summarize() Snowflake Cortex LLM function result:\n{summary_result.strip()}\n"
        # )

        completion_result = complete(user_text)
        print(
            f"Complete() Snowflake Cortex LLM function result:\n{completion_result.strip()}\n"
        )

        # answer_result = extract_answer(user_text)
        # print(
        #     f"ExtractAnswer() Snowflake Cortex LLM function result:\n{answer_result}\n"
        # )

        # sentiment_result = sentiment(user_text)
        # print(
        #     f"Sentiment() Snowflake Cortex LLM function result:\n{sentiment_result}\n"
        # )

        # translation_result = translate(user_text)
        # print(
        #     f"Translate() Snowflake Cortex LLM function result:\n{translation_result.strip()}\n"
        # )

    finally:
        if snowflake_session:
            # Close the Snowflake session
            snowflake_session.close()


if __name__ == "__main__":
    # Run the main function
    main()