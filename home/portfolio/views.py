from django.shortcuts import render,redirect
from datetime import datetime
from portfolio.models import new_contact
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import sys
import os

# Adds the parent directory to sys.path so 'src' is discoverable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
from src.retrieval import retrieval

# Create your views here.
def home(request):
    return render(request,'home.html')

def contact(request):
    if request.method=='POST':
        email=request.POST.get('email')
        topic=request.POST.get('topic')
        description=request.POST.get('description')
        date=datetime.today()
        obj=new_contact(
            email=email,
            topic=topic,
            description=description,
            date=date
        )
        obj.save()
        return redirect('submitcontact')
    return render(request,'contact.html')

@csrf_exempt
def chatbot(request):
    if request.method=='POST':
        data=json.loads(request.body)
        query=data.get('query')
        retriev_obj=retrieval()
        llm_response=retriev_obj.model_initialize(input_text=query)
        return JsonResponse({'response': llm_response})
    return render(request,'chatbot.html')

def submitcontact(request):
    return render(request,'submitcontact.html')