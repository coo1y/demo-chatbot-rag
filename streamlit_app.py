__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import MessagesPlaceholder

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
import chromadb
from langchain_community.vectorstores import Chroma

import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY_AN"]

from dotenv import load_dotenv
load_dotenv()

# app config
st.set_page_config(page_title="AI ตอบคำถามเอกสาร", page_icon="🤖")

chat_ai_icon = "icon/authoritative_government_officer.png"
chat_user_icon = "icon/user.png"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536,
)

# Initialize the Chroma client with the local storage path
chroma_client = chromadb.PersistentClient(path="vectordb/knowledge")

langchain_chroma = Chroma(client=chroma_client, collection_name='json', embedding_function=embeddings)

## ฟังก์ชั่นในการ retrieval chunk ที่เกี่ยวข้องกับ query จำนวน k chunk
def query_and_retrieve(query, k=3):
    base_retriever = langchain_chroma.as_retriever(search_kwargs={'k': k})
    retrieved_documents = base_retriever.invoke(query)
    # retrieved_metadatas = []
    relevant_passage = "\\n".join([
        doc.page_content + f" {doc.metadata}" 
        for doc in retrieved_documents
    ])

    return relevant_passage

def create_chain():
    # prompt จะประกอบด้วย relevant_passage, chat_history, query
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            You are a man working as an authoritative Thai government officer.
            Your job is to answer the following question only by using the context given below in the triple backticks, do not use any other information to answer the question.
            - If the answer is not in the context, just frankly say you can't answer that query because of no corresponding context.
            - If the answer is in the context, additionally show the document name concatenating with contextual key and subkey as the source of the referencing context below the answer in the italic format.
            
            Context: ```{relevant_passage}```"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])

    openai_model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        max_tokens=2000,
    )

    parser = StrOutputParser() #Set output >>StrOutputParser จะได้ออกมาเป็นข้อความ

    chain = prompt | openai_model | parser
    return chain

def get_response(query, chat_history):
    chain = create_chain()
    relevant_passage = query_and_retrieve(query, k=5)

    # print(f"""Debug
    #       - Question: {query}
    #       - Context: {relevant_passage}""")

    return chain.stream({
          "query" : query,
          "relevant_passage": relevant_passage,
          "chat_history": chat_history,
    })

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="""สวัสดีครับ ผมคือเจ้าหน้าที่ AI สำหรับการตอบคำถามจากเอกสารแบบอัตโนมัติ ผมสามารถตอบคำถามจากเอกสารเหล่านี้:
                  \n1. ระเบียบกระทรวงพาณิชย์ ว่าด้วยการอนุญาตให้ส่งผลิตภัณฑ์มันสำปะหลังออกไปนอกราชอาณาจักร สำหรับปี 2557 พ.ศ. 2556
                  \n2. ประกาศกระทรวงพาณิชย์ เรื่อง กำหนดด่านตุลกากรที่ผู้ส่ง หรือน่าสินค้ามาตรฐานแป้งมันสำปะหลังออกบอกราชอาณาจักรต้องแสดงใบรับรองมาตรฐานสินค้า พ.ศ. 2546
                  \n3. ประกาศสำนักงานมาตรฐานสินค้า เรื่อง กำหนดท้องที่หรือเขตตรวจสอบมาตรฐานสินค้าผลิตภัณฑ์มันสำปะหลังสำหรับส่งไปต่างประเทศ พ.ศ. 2562
                  \n4. ประกาศกระทรวงพาณิชย์ เรื่อง กำหนดให้แป้งมันสำปะหลังเป็นสินค้ามาตรฐานและมาตรฐานสินค้าแป้งมันสำปะหลัง พ.ศ. 2562
                  \n5. ประกาศกระทรวงพาณิชย์ เรื่อง กำหนดอัตราค่าบริการการตรวจสอบมาตรฐานสินค้าและการออกใบรับรองมาตรฐานสินค้าผลิตภัณฑ์มันสำปะหลัง
                  \n6. ประกาศกระทรวงพาณิชย์ เรื่อง หลักเกณฑ์และวิธีการการจัดให้มีการตรวจสอบมาตรฐานสินค้า และการตรวจสอบมาตรฐานสินค้าแป้งมันสําปะหลัง
                  \n7. ประกาศกระทรวงพาณิชย์ เรื่อง หลักเกณฑ์และวิธีการการจัดให้มีการตรวจสอบและการตรวจสอบมาตรฐานสินค้าผลิตภัณฑ์มันสำปะหลัง
                  \n8. ประกาศกระทรวงพาณิชย์ เรื่อง หลักเกณฑ์และวิธีการปฏิบัติเกี่ยวกับผลิตภัณฑ์มันสำปะหลังที่คุณภาพไม่เป็นไปตามมาตรฐานสินค้าที่กำหนด (ฉบับที่ 2)"""),
    ]

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar=chat_ai_icon):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar=chat_user_icon):
            st.write(message.content)

# user input
user_query = st.chat_input("พิมพ์คำถามตรงนี้ ...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human", avatar=chat_user_icon):
        st.markdown(user_query)

    with st.chat_message("AI", avatar=chat_ai_icon):
        response = get_response(user_query, st.session_state.chat_history[-6:])
        response = st.write_stream(response)

        st.session_state.chat_history.append(AIMessage(content=response))
