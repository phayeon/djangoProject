from django.db import models
from blog.blog_users.models import Blog_users


class Post(models.Model):
    use_in_migration = True
    post_id = models.AutoField(primary_key=True)
    title = models.TextField()
    content = models.TextField()
    create_at = models.TextField()
    updated_at = models.TextField()

    blog_user = models.ForeignKey(Blog_users, on_delete=models.CASCADE)

    class Meta:
        db_table = "blog_post"

    def __str__(self):
        return f'{self.pk} {self.title} {self.content} {self.create_at}' \
               f' {self.updated_at}'
