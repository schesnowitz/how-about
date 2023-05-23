# Generated by Django 4.2.1 on 2023-05-22 08:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('oai', '0008_rename_document_document_document_path'),
    ]

    operations = [
        migrations.CreateModel(
            name='DocumentQuery',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('query', models.CharField(blank=True, max_length=1000)),
                ('response', models.TextField(blank=True, null=True)),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
