from django.urls import path
from .views import ImageDetectionView, VideoDetectionView

urlpatterns = [
    path('detect/image/', ImageDetectionView.as_view(), name='detect-image'),
    path('detect/video/', VideoDetectionView.as_view(), name='detect-video'),
]