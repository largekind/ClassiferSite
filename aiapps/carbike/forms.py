from django import forms

class PhotoFrom(forms.Form):
  image = forms.ImageField(widget=forms.FileInput(attrs={'class':'custom-file-input'}))
