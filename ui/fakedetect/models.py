from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings

# Create your models here.

class CustomUser(AbstractUser):
    # 在这里可以添加自定义字段
    pass

class DetectionRecord(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    detection_time = models.DateTimeField(auto_now_add=True)
    result = models.TextField()

    def __str__(self):
        return f"{self.user.username} - {self.detection_time}"
