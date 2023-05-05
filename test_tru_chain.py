# from llama.hf import LLaMATokenizer

import torch
from langchain import LLMChain, PromptTemplate
from langchain.chains import (ConversationalRetrievalChain,
                              SimpleSequentialChain)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Pinecone
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from keys import *
from tru_chain import TruChain

class TestTruChain():

    def setup(self):
        print("setup")

        self.llm_model_id = "gpt2"
        # model_id = "decapoda-research/llama-7b-hf"
        # model_id = "decapoda-research/llama-13b-hf"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_id,
            device_map='auto',
            torch_dtype=torch.float16,
            local_files_only=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_id,
                                                       local_files_only=True)

        self.pipe = pipeline("text-generation",
                             model=self.model,
                             tokenizer=self.tokenizer,
                             max_new_tokens=16,
                             device_map="auto",
                             early_stopping=True)

        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    def test_qa_prompt(self):
        # llm = OpenAI()
        template = """Q: {question} A:"""
        prompt = PromptTemplate(template=template,
                                input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)

        tru_chain = TruChain(chain=llm_chain)

        tru_chain.run(dict(question="How are you?"))
        tru_chain.run(dict(question="How are you today?"))

        assert len(tru_chain.records) == 2

        assert tru_chain.model is not None

    def test_qa_db(self):
        index_name = "llmdemo"

        embedding = OpenAIEmbeddings(
            model='text-embedding-ada-002')  # 1536 dims

        docsearch = Pinecone.from_existing_index(index_name=index_name,
                                                 embedding=embedding)

        # llm = OpenAI(temperature=0,max_tokens=128)

        retriever = docsearch.as_retriever()
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm, retriever=retriever, return_source_documents=True)

        tru_chain = TruChain(chain)

    def test_sequential(self):
        template = """Q: {question} A:"""
        prompt = PromptTemplate(template=template,
                                input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)

        template_2 = """Reverse this sentence: {sentence}."""
        prompt_2 = PromptTemplate(template=template_2,
                                  input_variables=["sentence"])
        llm_chain_2 = LLMChain(prompt=prompt_2, llm=self.llm)

        seq_chain = SimpleSequentialChain(chains=[llm_chain, llm_chain_2],
                                          input_key="question",
                                          output_key="answer")
        seq_chain.run(
            question=
            "What is the average air speed velocity of a laden swallow?")

        tru_chain_2 = TruChain(seq_chain)

        # This run should not be recorded.
        seq_chain.run(
            question=
            "What is the average air speed velocity of a laden swallow? again")
        
        # These two should.
        tru_chain_2.run(
            question=
            "What is the average air speed velocity of a laden swallow?")
        tru_chain_2.run(
            question=
            "What is the average air speed velocity of a laden swallow?")

        assert len(tru_chain_2.records) == 2
        assert tru_chain_2.model is not None