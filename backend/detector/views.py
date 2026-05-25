from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from .ml_model import predict_image, predict_video


class ImageDetectionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        if 'image' not in request.FILES:
            return Response(
                {"error": "No image file provided"},
                status=status.HTTP_400_BAD_REQUEST
            )

        image_file = request.FILES['image']

        # Validate file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
        if image_file.content_type not in allowed_types:
            return Response(
                {"error": "Invalid file type. Please upload JPG, PNG or WEBP"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Run detection
        result = predict_image(image_file)

        if "error" in result:
            return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(result, status=status.HTTP_200_OK)


class VideoDetectionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        if 'video' not in request.FILES:
            return Response(
                {"error": "No video file provided"},
                status=status.HTTP_400_BAD_REQUEST
            )

        video_file = request.FILES['video']

        # Validate file type
        allowed_types = ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime']
        if video_file.content_type not in allowed_types:
            return Response(
                {"error": "Invalid file type. Please upload MP4, AVI or MOV"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Run detection
        result = predict_video(video_file)

        if "error" in result:
            return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(result, status=status.HTTP_200_OK)