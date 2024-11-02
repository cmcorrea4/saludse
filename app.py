import os
#from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import PyPDF2
from PIL import Image as Image, ImageOps as ImagOps
import glob
from gtts import gTTS
import os
import time
from streamlit_lottie import st_lottie
import json

try:
    os.mkdir("temp")
except:
    pass

with st.sidebar:
    st.subheader("Que es un sistema Experto?")
    st.write(
    """Este sistema experto te resolverá dudas sobre la conformadora de talones.
       te ayudará a aprender lo básico sobre la máquina.
       
    """
                )            

st.title('Sistema Experto CONFORMADORA DE TALONES💬')
image = Image.open('Instructor.png')
st.image(image)
#with open('Experts.json') as source:
#     animation=json.load(source)
#st.lottie(animation,width =350)

#ke = st.text_input('Ingresa tu Clave')
#os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['OPENAI_API_KEY'] = st.secrets["settings"]["key"] #ke

#st.write(st.secrets["settings"]["key"])

pdfFileObj = open('example2.pdf', 'rb')
 
# creating a pdf reader object
pdfReader = PyPDF2.PdfReader(pdfFileObj)


    # upload file
#pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

   # extract the text
#if pdf is not None:
from langchain.text_splitter import CharacterTextSplitter
 #pdf_reader = PdfReader(pdf)
pdf_reader  = PyPDF2.PdfReader(pdfFileObj)
text = ""
for page in pdf_reader.pages:
         text += page.extract_text()

   # split into chunks
text_splitter = CharacterTextSplitter(separator="\n",chunk_size=500,chunk_overlap=20,length_function=len)
chunks = text_splitter.split_text(text)

# create embeddings
embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_texts(chunks, embeddings)

# show user input
st.subheader("Usa el campo de texto para hacer tu pregunta")
user_question = st.text_area(" ")
if user_question:
        docs = knowledge_base.similarity_search(user_question)

        llm = OpenAI(model_name="gpt-4o-mini")
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
        st.write(response)

        def text_to_speech(text, tld):
                
                tts = gTTS(response,"es", tld , slow=False)
                try:
                    my_file_name = text[0:20]
                except:
                    my_file_name = "audio"
                tts.save(f"temp/{my_file_name}.mp3")
                return my_file_name, text

    
        if st.button("Escuchar"):
          result, output_text = text_to_speech(response, 'es-us')
          audio_file = open(f"temp/{result}.mp3", "rb")
          audio_bytes = audio_file.read()
          st.markdown(f"## Escucha:")
          st.audio(audio_bytes, format="audio/mp3", start_time=0)



            
          def remove_files(n):
                mp3_files = glob.glob("temp/*mp3")
                if len(mp3_files) != 0:
                    now = time.time()
                    n_days = n * 86400
                    for f in mp3_files:
                        if os.stat(f).st_mtime < now - n_days:
                            os.remove(f)
                            print("Deleted ", f)
            
            
          remove_files(7)



