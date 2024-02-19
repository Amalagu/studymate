from django import forms
from django.forms import formset_factory
from .models import Document

class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('file',)

""" class BaseDocumentFormSet(forms.BaseModelFormSet):
    def clean(self):
        # Custom formset validation logic, if needed
        pass """




