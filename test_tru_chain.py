# from llama.hf import LLaMATokenizer

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import LLMChain, PromptTemplate
from langchain.chains import SimpleSequentialChain
import torch
import pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from tru import TruChain

model_id = "gpt2"
# model_id = "decapoda-research/llama-7b-hf"
# model_id = "decapoda-research/llama-13b-hf"

model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16, local_files_only=True)#.to("cuda:0")
print("model loaded")
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
print("tokenizer loaded")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=16,
    device_map="auto",
    early_stopping=True
)
llm = HuggingFacePipeline(pipeline=pipe)

class TruChainTests():

    def test_qa_prompt():
        # llm = OpenAI()
        template = """Q: {question} A:"""
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        tru_chain = TruChain(chain=llm_chain)

        tru_chain.run(dict(question="How are you?"))
        tru_chain.run(dict(question="How are you today?"))

        tru_chain.records

        tru_chain.model

    def test_qa_db():
        index_name = "llmdemo"

        embedding = OpenAIEmbeddings(model='text-embedding-ada-002') # 1536 dims

        docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embedding)

        # llm = OpenAI(temperature=0,max_tokens=128)

        retriever = docsearch.as_retriever()
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True)
        
        tru_chain = TruChain(chain)

    def test_sequential():
        verb = False

        template = """Q: {question} A:"""
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=verb)

        template_2 = """Reverse this sentence: {sentence}."""
        prompt_2 = PromptTemplate(template=template_2, input_variables=["sentence"])
        llm_chain_2 = LLMChain(prompt=prompt_2, llm=llm, verbose=verb)

        # print(llm_chain.run(question="What is the average air speed velocity of a laden swallow?"))

        print(llm_chain_2.run(sentence="How are you doing?"))

        seq_chain = SimpleSequentialChain(chains=[llm_chain, llm_chain_2], input_key="question", output_key="answer")
        seq_chain.run(question="What is the average air speed velocity of a laden swallow?")

        tru_chain_2 = TruChain(seq_chain)

        seq_chain.run(question="What is the average air speed velocity of a laden swallow? again")
        tru_chain_2.run(question="What is the average air speed velocity of a laden swallow?")
        tru_chain_2.run(question="What is the average air speed velocity of a laden swallow?")

        tru_chain_2.records
        tru_chain_2.model