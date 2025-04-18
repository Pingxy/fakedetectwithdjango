import os
from celery import Celery

# 设置 Django 的 settings 模块
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ui.settings') # 确保 'ui.settings' 指向您的 settings 文件

app = Celery('ui') # 使用您的项目名

# 使用 Django settings 文件配置 Celery
app.config_from_object('django.conf:settings', namespace='CELERY')

# 自动发现各个 app 下的 tasks.py 文件
app.autodiscover_tasks()

@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
