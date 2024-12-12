import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from openai import OpenAI
import os

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def get_pdf_texts(pdf_docs):
    text =""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(raw_text)

    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(text_chunks, embeddings)

    return vectorstore


def get_conversation_chain(vectorstore):

    # 프롬프트에 페르소나 적용
    llm = ChatOpenAI(
        model_name = "gpt-4",
        temperature=0, 
        openai_api_key=OPENAI_API_KEY  # OpenAI API 키 입력
    )
    
    # PromptTemplate 생성
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"], # 필수 변수: 검색 결과(context)와 질문(query)
        template="""
            
        참고할 문서 내용:
            {context}
        사용자 질문:
            {question}
        답변:
        """
    )

    # RAG 시스템 (RetrievalQA 체인) 구축
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, # 언어 모델
        chain_type="stuff", # 검색된 모든 문서를 합쳐 전달 ("stuff" 방식)
        retriever=vectorstore.as_retriever(), # 벡터 스토어 리트리버
        return_source_documents=False, # 답변에 사용된 문서 출처 반환
        chain_type_kwargs={"prompt": custom_prompt}
    )

    return qa_chain
    
def getImageFromGpt(input_text):
    
    image_completion = client.images.generate(
        model="dall-e-3",
        prompt=input_text,
        size="1024x1024",
        n=1,
    )

    image_url = image_completion.data[0].url
    return image_url
    


st.title("📕📝🔍 팀 이름 및 설명, 로고 생성 서비스")
user_uploads = st.file_uploader("원하는 팀 이름 및 설명 양식을 업로드해주세요", accept_multiple_files=True)

if user_uploads :
    if st.button("예시 PDF 업로드"):
        with st.spinner("PDF 준비중입니다"):
            #load
            raw_text = get_pdf_texts(user_uploads)
            #split
            chunks = get_text_chunks(raw_text)
            #embed & store
            vectorestore = get_vectorstore(chunks)
            #chain
            st.session_state.converstaion = get_conversation_chain(vectorestore)            
            st.success("준비완료")


with st.form("프로젝트에 대해 설명해주세요."):
    domain = st.text_input("프로젝트 도메인을 알려주세요",placeholder="금융, 여행 등...")
    description = st.text_area("프로젝트에 대한 설명을 해주세요", placeholder="여행지에서 사람들과 대화할 수 있는 서비스야")
    submitted = st.form_submit_button("제출")

    if submitted:
        query = '프로젝트는 '+ domain +'에 관련된 프로젝트야.' + "자세한 설명은 다음과 같아" + description
            
        with st.spinner("답변 준비 중") :
            result = st.session_state.conversation.invoke({"query" : query})
            response = result['result']
        img_url = getImageFromGpt(query + "이 프로젝트를 진행하는 팀 로고를 5개 그려줘")
        st.write(response)
        st.image(img_url)