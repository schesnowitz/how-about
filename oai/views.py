from django.db.models import Q
from django.views.generic import ListView, DetailView
from .models import Oai, Document, DocumentPath, DocumentQuery
from django.core.files.storage import FileSystemStorage
from django.shortcuts import redirect, render
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.views.generic.edit import UpdateView, DeleteView
from django.urls import reverse_lazy

# langchain

from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader, YoutubeLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from datetime import datetime

dt = datetime.now()
from django import forms
from ckeditor.widgets import CKEditorWidget



class UpdateForm(forms.ModelForm):
    content = forms.CharField(widget=CKEditorWidget())

    class Meta:
        model = Oai
        fields = ["reporter", "title", "content"]


def update_oai(request, pk):
    form = UpdateForm()
    obj = Oai.objects.get(id=pk)

    form = UpdateForm(request.POST or None, instance=obj)
    if form.is_valid():
        oai = form.save(commit=False)
        oai.save()
        messages.success(request, "AI has saved your changes.")
        return redirect("oai:list")
    return render(request, "oai/oai_form.html", context={"pk": pk, "form": form})





class OaiDeleteView(DeleteView):
    model = Oai
    success_url = reverse_lazy("oai:list")


def create_oai(request):
    if request.method == "POST":
        try:
            if request.POST.get("form_type") == "formOne":
                url = request.POST["url"]
                # print(url)
                context = {"url": url}

                llm = OpenAI(temperature=0.9, verbose=True)
                you_tube = "youtube"
                if you_tube in url:
                    print("youtube")
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                else:
                    ("not youtube")
                    loader = WebBaseLoader(web_path=url)
                data = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000, chunk_overlap=250, length_function=len
                )

                docs = text_splitter.split_documents(data)

                embeddings = OpenAIEmbeddings()
                db = FAISS.from_documents(docs, embeddings)
                db.add_documents(docs)

                """
                -------------------------------------------------------------
                Reporter Name
                -------------------------------------------------------------
                """

                prompt_template = """Use the context below to create a ficticious name for a news reporter.
                the name should be less than 25 characters, only generate a full name:
                    name: {name}
                    context: {query}
                    Reporter Name:"""

                story_reporter = PromptTemplate(
                    template=prompt_template, input_variables=["name", "query"]
                )

                chain = LLMChain(llm=llm, prompt=story_reporter, verbose=True)

                query = "write a title for the story"
                docs = db.similarity_search(query, k=3)

                story_reporter_name = chain.run({"name": docs, "query": query})
                print(story_reporter_name)
                # num_tokens = llm.get_num_tokens(prompt_template)
                # print (f"Our prompt has {num_tokens} tokens")

                """
                -------------------------------------------------------------
                Story Title
                -------------------------------------------------------------
                """

                prompt_template = """Use the context below to write  a title.:
                    Context: {title}
                    Topic: {query}
                    Blog title:"""

                title_content = PromptTemplate(
                    template=prompt_template, input_variables=["title", "query"]
                )

                chain = LLMChain(llm=llm, prompt=title_content, verbose=False)

                query = "write a title for the story"
                docs = db.similarity_search(query, k=3)
                story_title = chain.run({"title": docs, "query": query})
                print(story_title)
                # callback_token_getter()
                """
                -------------------------------------------------------------
                Story Content
                -------------------------------------------------------------
                """

                prompt_template = """Use the context below to write a 700 word blog post 
                about the context below make sure the story ends with a complete sentence:
                    Context: {context}
                    Topic: {query}
                    Blog post:"""

                prompt_content = PromptTemplate(
                    template=prompt_template, input_variables=["context", "query"]
                )

                chain = LLMChain(llm=llm, prompt=prompt_content, verbose=False)

                query = "write a detailed synopsis of this story"
                docs = db.similarity_search(query, k=3)
                story_content = chain.run({"context": docs, "query": query})
                print(story_content)
                # callback_token_getter()

                context = {
                    "url": url,
                    "story_content": story_content,
                    "story_reporter_name": story_reporter_name,
                    "story_title": story_title,
                }

                return render(
                    request=request, template_name="oai/create.html", context=context
                )

            if (
                request.POST.get("form_type") == "formTwo"
                and request.POST.get("story_title")
                and request.POST.get("story_content")
                and request.POST.get("url")
                and request.POST.get("story_reporter_name")
            ):
                oai = Oai()
                oai.title = request.POST.get("story_title")
                oai.content = request.POST.get("story_content")
                oai.url = request.POST.get("url")
                oai.reporter = request.POST.get("story_reporter_name")
                oai.save()
                print(oai.id)
                messages.success(request, "AI request completed.")

                return redirect("oai:update", oai.id)
            else:
                return render(request, "oai/create.html")

        except:
            messages.success(request, "There was a problem with the URL.")
            return render(request, "oai/create.html", {})

    return render(request, "oai/create.html", {})


class SearchResultsView(ListView):
    model = Oai
    template_name = "oai/search.html"

    def get_queryset(self):
        query = self.request.GET.get("q")
        if query == "":
            messages.add_message(
                self.request,
                messages.INFO,
                "Your search was blank, please enter a valid search request.",
            )
            redirect("/")
        else:
            object_list = Oai.objects.filter(
                Q(content__icontains=query) | Q(title__icontains=query)
            )

            if not object_list.exists():
                messages.add_message(
                    self.request,
                    messages.INFO,
                    "Your search did not return any results.",
                )
            else:
                # print(object_list)
                return object_list


# def search_posts(request):
#     if request.method == Post:
#         search_ai = request.POST('search_ai')
#         context = {'search_ai' : search_ai}
#         print(context)
#         return render(request, 'aipost/search.html', context=context)
#     else:
#         context = {}
#         return render(request, 'aipost/search.html', context=context)


class OaiListView(ListView):
    model = Oai
    template_name = "oai/list.html"
    paginate_by = 10
    ordering = ["-created_on"]

    # def get_context_data(self, **kwargs):
    #     context = super(OaiListView, self).get_context_data(**kwargs)
    #     context["side_page"] = Oai.objects.all().order_by("?")[:6]

    #     return context


class OaiDetailView(DetailView):
    model = Oai
    template_name = "oai/detail.html"
 



class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('description', 'document_path', 'document_url' )
        widgets = {'document_url': forms.HiddenInput()}
        def __init__(self, *args, **kwargs):
            super(DocumentForm, self).__init__(*args, **kwargs)
            self.fields['document_url'].initial = self.document_path
def document_upload(request):   
    user_docs = Document.objects.order_by('-uploaded_at')
    if request.method == 'POST':
        user_docs = Document.objects.order_by('-uploaded_at')
        form = DocumentForm(request.POST, request.FILES)

        if form.is_valid():
            # f = form.save(commit=False)
            # document_url = form.cleaned_data.get("document_path")
            # f.document_url = document_url
     
            # f.save()


            initial_obj = form.save(commit=False)
            initial_obj.save()   
            # return path name from `upload_to='documents/'` in your `models.py` + absolute path of file.
            # eg; `documents/filename.csv`
            print(initial_obj.document_path)

            # return `MEDIA_URL` + `upload_to` + absolute path of file.
            # eg; `/media/documents/filename.csv`
            print(initial_obj.document_path.url)
  
            form.save()


            return render(request, 'oai/document_upload.html', {'form': form, 'user_docs' : user_docs})
        else:
            form = DocumentForm()
            return render(request, 'oai/document_upload.html', {'form': form, 'user_docs' : user_docs})
   
  
    return render(request, 'oai/document_upload.html', context={'user_docs' : user_docs})

class DocumentSelectForm(forms.ModelForm):
    class Meta:
        model = DocumentPath
        fields = ('doc_path',)


def documents(request):   
    obj = Document.objects.all()
    last_doc_object = DocumentPath.objects.last()
    if request.method == 'POST':
        obj = Document.objects.all()
        form = DocumentSelectForm(request.POST)
    
        if form.is_valid():
            form.save() 

        else:
            form = DocumentSelectForm() 

    return render(request, 'oai/documents.html', {'obj': obj, 'form' : form, 'last_doc' : last_doc_object})
  
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA



def document_interact(request, pk):
    doc = Document.objects.get(pk=pk) 
    if request.method == "POST":

  
        llm = OpenAI(temperature=0.1, verbose=True)
        file_url = request.build_absolute_uri(doc.document_path.url)

        loader = PyPDFLoader(file_url) 
        documents = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_documents(texts, embeddings)


        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=docsearch.as_retriever())

 

        query = request.POST.get("doc_query")
        response = qa.run(query)
                

        return render(request, 'oai/doc_interact.html', {'doc' : doc, 'response' : response})
    return render(request, 'oai/doc_interact.html', {'doc' : doc})
        

            #     if (
            #     request.POST.get("form_type") == "formTwo"
                
            #     and request.POST.get("story_content")
            #     and request.POST.get("url")
            #     and request.POST.get("story_reporter_name")
            # ):
            #     oai = Oai()
            #     oai.title = request.POST.get("story_title")

