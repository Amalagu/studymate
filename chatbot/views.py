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

from django.views.decorators.csrf import csrf_protect, csrf_exempt 
#####################################
#   INITIAL SETUP
#####################################


BASE_DIR = Path(__file__).resolve().parent.parent
faiss_directory = os.path.join(BASE_DIR, 'my_vector_index.faiss')
vector_database_dir = os.path.join(BASE_DIR, 'vector-databases')

SUCCESSFUL_CONNECTION=True;
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")


#Warning: model not found. Using cl100k_base encoding
    #PASSING THE ABOVE TEXT TO CHATGPT ALWAYS CAUSES AN ERROR


#####################################################
#   ACCOUNT ROUTES
#####################################################


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

@csrf_protect
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
                return redirect('upload_file')
            except Exception as e:
                print(str(e))
                error_message = 'Error creating account'
                return render(request, 'register.html', {'error_message': error_message})
        else:
            error_message = 'Password do not match'
            return render(request, 'register.html', {'error_message': error_message})
    return render(request, 'register.html')


@csrf_protect
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


@csrf_protect
def logout(request):
    auth.logout(request)
    return redirect('login')


##############################################################
# DOCUMENT UPLOADING AND VECTORIZATION CODES
##############################################################

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


@login_required
def upload_file(request):
    DocumentFormSet = formset_factory(DocumentForm,  extra=2)  # Adjust 'extra' as needed
    user = request.user
    if request.method == 'POST':
        formset = DocumentFormSet(request.POST, request.FILES)
        if formset.is_valid():
            newchat = ChatModel.objects.create(user=user, conversation_history={})
            for form in formset:
                if form.cleaned_data.get('file'):
                    uploaded_file: UploadedFile = form.cleaned_data['file']
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    # Check if the uploaded file is a PDF or MS Word file
                    if file_extension not in ['pdf', 'doc', 'docx']:
                        error_message = 'Please upload only a PDF or MS Word file!'
                        try:
                            newchat.delete()
                        except:
                            pass
                        return render(request, 'upload.html', {'formset': DocumentFormSet(), 'error_message': error_message})
                    
                    document = form.save(commit=False)
                    vector_save_path = f"{vector_database_dir}/{user.username}/{uploaded_file.name}"
                    document.filepath = vector_save_path
                    document.chat = newchat
                    document.save()

                    if file_extension == 'pdf':
                        chunklist = get_pdf_text(f"{BASE_DIR}{document.file.url}")
                    else:
                        chunklist = get_word_text(f"{BASE_DIR}{document.file.url}")
                    
                    vectorstore_db = FAISS.from_documents(chunklist, embeddings)
                    vectorstore_db.save_local(vector_save_path)

            oldchats = ChatModel.objects.filter(user=user).order_by('-created_at').exclude(id=newchat.id)
            oldchats.delete()
            return redirect('chatbot')
            #document_url = request.build_absolute_uri(document.file.url)
            #print("THIS IS REQUEST DOCUMENT URL ", document_url) #http://127.0.0.1:8000/media/uploads/Screenshot_1_wcSGcWy.png
            #print("THIS IS REQUEST DOCUMENT URL ", document.file.url) #/media/uploads/Screenshot_2_huTU50u.png
    else:
        formset = DocumentFormSet()
    
    return render(request, 'upload.html', {'formset': formset})




############################################################################
#       CHATBOT CODES
############################################################################


""" def format_conversation_history(conversation_history):
    formatted_history = [{'role': 'system', 'content': "You are a note reading assistant"},]
    for prompt, response in conversation_history.items():
        tempdict = {}
        if 'user_msg' in prompt.split():
            tempdict['role'] = 'user'
        else:
            tempdict['role'] = 'assistant'
        tempdict['content'] = response
        formatted_history.append(tempdict)
    return formatted_history """


def format_conversation_history(conversation_history):
    """
    Format the conversation history to be passed to default_query parameter.
    """
    formatted_history = [{'role': 'system', 'content': "You are a note reading assistant"},]
    chatconversation_items = list(conversation_history.items())
    if len(chatconversation_items) >= 4:
        last_entries = chatconversation_items[-4:]
    elif len(chatconversation_items) >= 2:
        last_entries = chatconversation_items[-2:]
    else:
        last_entries = []
        return []
    for prompt, response in last_entries:
        tempdict = {}
        if 'user_msg' in prompt.split():
            tempdict['role'] = 'user'
        else:
            tempdict['role'] = 'assistant'
        tempdict['content'] = response
        formatted_history.append(tempdict)
    return formatted_history

messages= [
    {'role': 'system', 'content': "You are a note reading assistant"},
    {'role': 'user', 'content': "Hello"},
    {'role': 'assistant', 'content': "I am fine"},
    {'role': 'user', 'content': "Hello"},
    {'role': 'assistant', 'content': "I am fine"},
]

def get_response_from_student_data(user, query, k=2):
    search_results = {}
    try:
        user_chat = ChatModel.objects.get(user=user)
        documents = user_chat.documents.all() 
    except ChatModel.DoesNotExist:
        print("CHAT DOES NOT EXIT")
        pass
    
    for doc in documents:
        filename = doc.file.name.split('/')[-1].lower()
        db = FAISS.load_local(doc.filepath, embeddings)
        docs = db.similarity_search(query, k)
        search_results[filename]  = " ".join([d.page_content for d in docs])
    
    conversation_history = format_conversation_history(user_chat.conversation_history)
    """ chatconversation_items = list(user_chat.conversation_history.items())
    if len(chatconversation_items) >= 4:
        last_entries = chatconversation_items[-4:]
    elif len(chatconversation_items) >= 2:
        last_entries = chatconversation_items[-2:]
    else:
        last_entries = []
    for index, (prompt_key, response_key) in enumerate(last_entries, start=1):
        tempdict={}
        if 'user_msg' in prompt_key.split():
            tempdict['role'] = 'user'
        else:
            tempdict['role'] = 'assistant'
        tempdict['content'] = response_key
        conversation_history.append(tempdict) """
    

    system_message = f"""
    Task Definition:
    As a student reading assistant, your primary goal is to help students comprehend their lecture notes effectively. 
    You'll achieve this by providing concise yet comprehensive explanations, examples, and analogies when necessary.

    Task Details:
    Your task is to respond to user questions using your knowledge base with the provided lecture notes as context. The Conversation history gives you
    context and  insight into your recent interation with the user. Your responses should be 
    brief, comprehensive, and directly related to the documents provided. If a question falls outside the scope of 
    the provided context and the context of the provided recent conversation history,
    respond with "This Question Seems to Be Out of the Scope of Your Lecture Notes." however you can
    always respond to greetings and pleasantries.

    Template Variables:
    - User: {user} (This variable represents the user's name or identifier.)
    - Question: {query} (The question posed by the user.)
    - Documents: {search_results} (List of documents from the lecture notes used as context.)
    - Conversation: {conversation_history} (A dictionary of recent history past conversations with the user)

    Remember, your role is to assist students in understanding their study materials better using your knowledge base with the provided lecture notes as context 
    (you only refer or call their names when necessary). Good luck!
    """


    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                    max_tokens=250,
                    #default_query= system_message
                    )
    prompt = PromptTemplate(
        input_variables=["question", "docs", "user", "conversation_history"],
        template="""
    Task Definition:
    As a student reading assistant, your primary goal is to help students comprehend their lecture notes effectively. 
    You'll achieve this by providing concise yet comprehensive explanations, examples, and analogies when necessary.

    Task Details:
    Your task is to CONTINUE THIS CONVERSATION: {conversation_history} AND respond to user questions using your knowledge base with the provided lecture notes as context. The Conversation history gives you
    context and  insight into your recent interation with the user. Your responses should be 
    brief, comprehensive, and directly related to the documents provided. If a question falls outside the scope of 
    the provided context and the context of the provided recent conversation history,
    respond with "This Question Seems to Be Out of the Scope of Your Lecture Notes." however you can
    always respond to greetings and pleasantries.

    Template Variables:
    - User: {user} (This variable represents the user's name or identifier.)
    - Question: {question} (The question posed by the user.)
    - Documents: {docs} (List of documents from the lecture notes used as context.)
    - Conversation:  (A dictionary of recent history past conversations with the user)

    Remember, your role is to assist students in understanding their study materials better using your knowledge base with the provided lecture notes as context 
    (you only refer or call their names when necessary). Good luck!
    """,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({'question':query, 'docs': search_results, 'user': user.username, "conversation_history": conversation_history})
    response = response['text']
    return response, docs





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
        #print(docs)
        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chat.html', {'chats': chats})






"""
        You are a students' reading assistant who helps them comprehend their notes better,
        by giving them brief and comprehensive explanations (with examples and even analogies ) where necessary.
        Answer the following question from me: {question}
        By only using the following documents (from my lecture notes) as context: {docs}
        Your response should be brief, comprehensive and should only expand more on the document when requested in the question. If the question
        is not related to the provided context, simply respond with "This Question Seem To Be Out Of the Scope 
        Of Your Lecture Note".
        My name is {user}, but refer to me only when necessary.
        """




##############################
#
#       OLD CODE SAMPLES 
#
#################################

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




""" @login_required
def create_signup_chat(request):
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
            return HttpResponse("AN ERROR OCCURED!!")
 """



""" def get_word_text_chunks(text):
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
    return vectorstore """