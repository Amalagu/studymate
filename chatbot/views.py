from django.utils import timezone
from django.http import JsonResponse
from django.shortcuts import render, redirect
import openai
from django.contrib import auth
from django.contrib.auth.models import User
from .models import Chat
from .models import ChatModel

from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
load_dotenv()

""" import openai
from dotenv import dotenv_values

config = dotenv_values('.env')
openai.api_key = config['OPENAI_API_KEY'] """


embeddings = OpenAIEmbeddings()
# embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

loader = Docx2txtLoader("./handbook.docx")
handbook_doc = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs_structure = text_splitter.split_documents(handbook_doc)
vectorstore_db = FAISS.from_documents(docs_structure, embeddings)





def get_response_from_query(db, query, k=4):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        YouR name is Mr. Vitalis, you are a course adviser to the final year (500 level) students
        of the Federal University of Technology. 
        Please act the role and refer any enquiry that you are not sure of to the Departments Admin Officer (DAO).
        
        Answer the following question: {question}
        By searching the following document: {docs}
        
        Only use the factual information from the word document to answer the question
        while noting that the docment is correct and up to date.
        
        If you feel like you don't have enough information to answer the question, refer the user to the Department's Admin Officer (DAO)".
        
        
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.invoke({'question':query, 'docs':docs_page_content})
    response = response['text']
    #response = response.replace("\n", "")
    return response, docs










""" def chatbot(request):
    chats = Chat.objects.filter(user=request.user)

    if request.method == 'POST':
        message = request.POST.get('message')
        response, docs = get_response_from_query(vectorstore_db, message)

        chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
        chat.save()
        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chatbot.html', {'chats': chats})

 """

def chatbot(request):
    chats = ChatModel.objects.filter(user=request.user).order_by('-created_at').first()

    if request.method == 'POST':
        message = request.POST.get('message')
        response, docs = get_response_from_query(vectorstore_db, message)

        if chats:
            # If there is an existing chat, update the conversation history
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
                auth.login(request, user)
                return redirect('chatbot')
            except:
                error_message = 'Error creating account'
                return render(request, 'register.html', {'error_message': error_message})
        else:
            error_message = 'Password dont match'
            return render(request, 'register.html', {'error_message': error_message})
    return render(request, 'register.html')

def logout(request):
    auth.logout(request)
    return redirect('login')
