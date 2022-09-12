from multiprocessing import context
from django.shortcuts import render , redirect
from django.http import HttpResponse
from django.template import loader
from torch import torch
from .forms import PhotoForm
from .models import Photo

def index(request):
  template = loader.get_template('carbike/index.html')
  context = {'form':PhotoForm()}
  return HttpResponse(template.render(context, request))

def predict(request):
  #POST外は無効
  if not request.method == 'POST':
    return redirect('carbike:index')

  form = PhotoForm(request.POST, request.FILES)
  if not form.is_valid():
    raise ValueError('Formが無効です')

  photo = Photo(image = form.cleaned_data['image'])
  predicted, percent = photo.predict()

  template = loader.get_template('carbike/result.html')
  print(template)

  context = {
    'predicted' : predicted,
    'percentage' : torch.max(percent).item() * 100
  }
  #print(template.render(context, request))
  return HttpResponse(template.render(context, request))