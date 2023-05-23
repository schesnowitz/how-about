from django.db import models

from ckeditor.fields import RichTextField
 
class Oai(models.Model):
    title = models.CharField(max_length=2000, blank=True, null=True)
    content = RichTextField(blank=True, null=True)
    url = models.URLField(blank=True, null=True, max_length=3000)
    created_on = models.DateTimeField(auto_now_add=True, blank=True, null=True)
    updated_on = models.DateTimeField(auto_now=True, blank=True, null=True)
    reporter = models.CharField(max_length=1000, blank=True, null=True)
    def __str__(self):
        return self.title


class Document(models.Model):
    description = models.CharField(max_length=255, blank=True)
    document_path = models.FileField(upload_to='documents/')
    document_url = models.URLField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

class DocumentPath(models.Model):
    doc_path = models.CharField(max_length=1000, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

class DocumentQuery(models.Model):
    query = models.CharField(max_length=1000, blank=True)
    response = models.TextField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)    