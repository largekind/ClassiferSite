from multiprocessing import context
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .forms import PhotoFrom

def index(request):
  template = loader.get_template('carbike/index.html')
  context = {'form':PhotoFrom()}
  return HttpResponse(template.render(context, request))

def predict(request):
  return HttpResponse("show predict")