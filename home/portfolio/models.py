from django.db import models

# Create your models here.
class new_contact(models.Model):
    email=models.EmailField(max_length=100)
    topic=models.CharField(max_length=50)
    description=models.CharField(max_length=500)
    date=models.DateTimeField()