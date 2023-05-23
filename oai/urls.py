from django.urls import path, include
from .views import (
    create_oai, 
    OaiDetailView, 
    OaiListView, 
    update_oai, 
    OaiDeleteView, 
    document_upload, 
    documents, 
    document_interact
)
app_name='oai'

urlpatterns = [
    path('create/', create_oai, name='create'),
    path("detail/<int:pk>/", OaiDetailView.as_view(), name="detail"),
    path("update/<int:pk>/", update_oai, name="update"),
    path("delete/<int:pk>/", OaiDeleteView.as_view(), name="delete"),
    path('list/', OaiListView.as_view(), name="list"),
    path('document_upload/', document_upload, name="doc_upload"),
    path('documents/', documents, name="documents"),
    path('document_interaction/<int:pk>/', document_interact, name="document_interact"),
]
