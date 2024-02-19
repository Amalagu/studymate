from django.db import models
from django.contrib.auth.models import User
from django.db.models import JSONField
import os
# Create your models here.
class Chat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username}: {self.message}'



class ChatModel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    conversation_history = JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username}: Conversation'


class Document(models.Model):
    chat = models.ForeignKey(ChatModel, related_name='documents', null=True, blank=True, on_delete=models.CASCADE)
    file = models.FileField(upload_to='uploads/')
    filepath = models.TextField(max_length=200, blank=True, null=True)

    def delete(self, *args, **kwargs):
        # Delete the associated file from the file manager
        if self.filepath:
            try:
                file_path = os.path.normpath(self.filepath)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print("FILE DELETED EXISTS")
            except Exception as e:
                print(str(e))
                pass

        # Call the parent class's delete method to delete the model instance
        super().delete(*args, **kwargs)
