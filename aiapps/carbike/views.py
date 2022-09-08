from django.shortcuts import render
from django.http import HttpResponse

def index(request):
  return HttpResponse("HelloWorld")

def predict(request):
  return HttpResponse("show predict")