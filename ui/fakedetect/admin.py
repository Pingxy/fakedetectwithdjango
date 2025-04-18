from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import DetectionRecord, CustomUser

# 自定义 CustomUser 的 Admin 显示
class CustomUserAdmin(UserAdmin):
    # 在列表页显示的字段
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff', 'date_joined')
    # 搜索字段
    search_fields = ('username', 'email', 'first_name', 'last_name')
    # 过滤选项
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'groups')
    # 可以在这里进一步定制编辑页面的字段布局 (fieldsets) 或其他选项
    # 默认的 UserAdmin fieldsets 通常已经很好了，所以这里暂时不覆盖

# Register your models here.
admin.site.register(DetectionRecord)
# 使用自定义的 Admin 类注册 CustomUser
admin.site.register(CustomUser, CustomUserAdmin)
