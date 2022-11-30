from django.db import models

from blog.blog_users.models import Blog_users
from blog.posts.models import Post


class Views(models.Model):
    use_in_migration = True
    view_id = models.AutoField(primary_key=True)
    ip_address = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    post = models.ForeignKey(Post, on_delete=models.CASCADE)
    blog_user = models.ForeignKey(Blog_users, on_delete=models.CASCADE)

    class Meta:
        db_table = "blog_views"

    def __str__(self):
        return f'{self.pk} {self.ip_address} {self.created_at}'
