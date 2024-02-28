from functools import wraps
from django.shortcuts import redirect
from django.contrib import messages
from .models import ChatModel

def document_required(view_func):
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        # Check if the user has an associated ChatModel with one or more documents
        if not ChatModel.objects.filter(user=request.user, documents__isnull=False).exists():
            # If not, redirect them or display a message
            messages.error(request, 'You must have at least one document uploaded to access this page.')
            return redirect('upload_file')  # Change 'your_redirect_view' to your desired redirect URL

        # If the user has an associated ChatModel with one or more documents, proceed to the view
        return view_func(request, *args, **kwargs)

    return _wrapped_view
