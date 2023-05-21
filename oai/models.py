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