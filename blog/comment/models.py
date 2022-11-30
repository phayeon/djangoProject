from django.db import models

from blog.blog_users.models import Blog_users
from blog.posts.models import Post


class Comment(models.Model):
    use_in_migration = True
    comment_id = models.AutoField(primary_key=True)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    parents_id = models.TextField(null=True)

    blog_user = models.ForeignKey(Blog_users, on_delete=models.CASCADE)
    post = models.ForeignKey(Post, on_delete=models.CASCADE)

    class Meta:
        db_table = "blog_comment"

    def __str__(self):
        return f'{self.pk} {self.content} {self.created_at} {self.updated_at}'
