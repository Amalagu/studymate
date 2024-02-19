from django.utils import timezone
from django.http import JsonResponse
from django.shortcuts import render, redirect, HttpResponse
from django.contrib.auth.decorators import login_required
from django.core.files.uploadedfile import UploadedFile
import openai
from django.contrib import auth, messages
from django.contrib.auth.models import User
from .models import Chat, ChatModel, Document
from .forms import DocumentForm
from django.forms import formset_factory


from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
load_dotenv()
import faiss
from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent
faiss_directory = os.path.join(BASE_DIR, 'my_vector_index.faiss')
vector_database_dir = os.path.join(BASE_DIR, 'vector-databases')

SUCCESSFUL_CONNECTION=True;

try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
except ConnectionError as e:
    SUCCESSFUL_CONNECTION = False
    print(f"Connection error: {e}")



#Warning: model not found. Using cl100k_base encoding
    #PASSING THE ABOVE TEXT TO CHATGPT ALWAYS CAUSES AN ERROR





""" 
loader = Docx2txtLoader("./handbook.docx")
handbook_doc = loader.load()

#vectorstore_db = faiss.read_index(faiss_directory) # load index from disk
vectorstore_db = faiss.IndexFlatL2(1024)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs_structure = text_splitter.split_documents(handbook_doc)
vectorstore_db = FAISS.from_documents(docs_structure, embeddings)
vectorstore_db.save_local(faiss_directory) """
#vectorstore_db = FAISS.load_local(faiss_directory, embeddings)




def get_pdf_text(pdf_docs):
    """loader = PyPDFLoader("example_data/layout-parser-paper.pdf")"""
    loader = PyPDFLoader(pdf_docs)
    pages = loader.load_and_split()
    return pages


def get_word_text(word_docs):
    loader = Docx2txtLoader(word_docs)
    doc_data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text = text_splitter.split_documents(doc_data)
    return text


def create_chat(request):
    if request.method == 'POST':
        try:
            oldchats = ChatModel.objects.filter(user=request.user)
            if oldchats:
                oldchats.delete()
            chat = ChatModel(
                            user=request.user, 
                            conversation_history={}
                        )
            chat.save()
            
            messages.success(request, 'New chat created successfully.')
            return redirect('upload_file')
        except Exception as e:
            print(e)
            messages.error(request, f'An error occurred: {str(e)}')
            return redirect('create_chat')
    return render(request, 'create_chat.html')



@login_required
def upload_file(request):
    chats = ChatModel.objects.filter(user=request.user).order_by('-created_at').first()
    DocumentFormSet = formset_factory(DocumentForm,  extra=2)  # Adjust 'extra' as needed
    user = request.user
    if request.method == 'POST':
        formset = DocumentFormSet(request.POST, request.FILES)
        if formset.is_valid():
            for form in formset:
                if form.cleaned_data.get('file'):
                    document = form.save(commit=False)
                    uploaded_file: UploadedFile = form.cleaned_data['file']
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    vector_save_path = f"{vector_database_dir}/{user.username}/{uploaded_file.name}"
                    document.filepath = vector_save_path
                    document.chat = chats
                    document.save()

                    
                    # Check if the uploaded file is a PDF or MS Word file
                    if file_extension == 'pdf':
                        chunklist= get_pdf_text(f"{BASE_DIR}{document.file.url}")
                    elif file_extension in ['doc', 'docx']:
                        chunklist = get_word_text(f"{BASE_DIR}{document.file.url}")  
                    else:
                        document.delete()
                        error_message = 'Please Upload only a pdf or MsWord File!'
                        return render(request, 'upload.html', { 'formset': DocumentFormSet(), 'error_message': error_message})
                    
                    vectorstore_db = FAISS.from_documents(chunklist, embeddings)
                    vectorstore_db.save_local(vector_save_path)
                    
                    #document_url = request.build_absolute_uri(document.file.url)
                    #print("THIS IS REQUEST DOCUMENT URL ", document_url) #http://127.0.0.1:8000/media/uploads/Screenshot_1_wcSGcWy.png
                    #print("THIS IS REQUEST DOCUMENT URL ", document.file.url) #/media/uploads/Screenshot_2_huTU50u.png
            return redirect('chatbot') 
    else:
        formset = DocumentFormSet()
    return render(request, 'upload.html', { 'formset': formset})















""" 
try:
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
except ConnectionError as e:
    SUCCESSFUL_CONNECTION = False
    print(f"Connection error: {e}")

loader = Docx2txtLoader("./handbook.docx")
handbook_doc = loader.load()

chunk_size = 1000
chunk_overlap = 100

text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunks = text_splitter.split_documents(handbook_doc)

# Initialize an empty FAISS index
vectorstore_db = None

for chunk in chunks:
    # Generate embeddings for the current chunk
    chunk_embeddings = embeddings.encode(chunk)

    # Create a new FAISS index or append to an existing one
    if vectorstore_db is None:
        vectorstore_db = FAISS.from_embeddings(chunk_embeddings)
    else:
        vectorstore_db.add_embeddings(chunk_embeddings)

# Save the entire index to a file
faiss.write_index(vectorstore_db, "my_vector_index.faiss")
 """








def get_response_from_student_data(user, query, k=2):
    search_results = {}
    try:
        user_chat = ChatModel.objects.get(user=user)
        documents = user_chat.documents.all() 
    except ChatModel.DoesNotExist:
        pass
    
    for doc in documents:
        filename = doc.file.name.split('/')[-1].lower()
        db = FAISS.load_local(doc.filepath, embeddings)
        docs = db.similarity_search(query, k)
        search_results[filename]  = " ".join([d.page_content for d in docs])
        
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=250)
    prompt = PromptTemplate(
        input_variables=["question", "docs", "user"],
        template="""
        You are a students' reading assistant who helps them comprehend their notes better,
        by giving them brief and comprehensive explanations (with examples and even analogies ) where necessary.
        Answer the following question from me: {question}
        By only using the following documents (from my lecture notes) as context: {docs}
        Your response should be brief, comprehensive and should only expand more on the document when requested in the question. If the question
        is not related to the provided context, simply respond with "This Question Seem To Be Out Of the Scope 
        Of Your Lecture Note".
        My name is {user}, but refer to me only when necessary
        """,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({'question':query, 'docs': search_results, 'user': user.username})
    response = response['text']
    return response, docs


def get_word_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators='\n',
        chunk_size=1000, 
        chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks
    #chunks = text_splitter.split_documents(handbook_doc)


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore



@login_required
def chatbot(request):
    chats = ChatModel.objects.filter(user=request.user).order_by('-created_at').first()


    if request.method == 'POST':
        message = request.POST.get('message')
        response, docs = get_response_from_student_data(request.user, message)
        if chats:
            chat_history = chats.conversation_history
            chat_history[f'user_msg {len(chat_history) + 1}'] = message
            chat_history[f'bot_response {len(chat_history)}'] = response
            chats.save()
        else:
            # If no existing chat, create a new one
            chat_history = {f'user_msg 1': message, f'bot_response 1': response}
            new_chat = ChatModel(user=request.user, conversation_history=chat_history)
            new_chat.save()
        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chatbot.html', {'chats': chats})







def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('chatbot')
        else:
            error_message = 'Invalid username or password'
            return render(request, 'login.html', {'error_message': error_message})
    else:
        return render(request, 'login.html')



def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 == password2:
            try:
                user = User.objects.create_user(username, email, password1)
                user.save()
                if not os.path.exists(f"{vector_database_dir}/{user.username}"):
                    os.makedirs(f"{vector_database_dir}/{user.username}")
                auth.login(request, user)
                return redirect('create_chat')
            except Exception as e:
                print(str(e))
                error_message = 'Error creating account'
                return render(request, 'register.html', {'error_message': error_message})
        else:
            error_message = 'Password dont match'
            return render(request, 'register.html', {'error_message': error_message})
    return render(request, 'register.html')



def logout(request):
    auth.logout(request)
    return redirect('login')
