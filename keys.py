import os
# piotrm openai but shared with truera:
os.environ['OPENAI_API_KEY']='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' # llmdemo-piotrm

# piotrm pinecone:
PINECONE_API_KEY='781d8e2d-9220-4048-b041-b59508077e77' # llmdemo, 1536 dim cosine
PINECONE_ENV='us-west1-gcp-free'

# piotrm huggingface:
HUGGINGFACE_API_KEY='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'
HUGGINGFACE_HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# piotrm slack / TruBot app:
SLACK_TOKEN = "xoxb-938974664130-5194283342932-hTjjCuMrjGti2ADOLsFYllpB"
SLACK_SIGNING_SECRET = "498d1de2b6eacd4fabdfae4e4394f097"
