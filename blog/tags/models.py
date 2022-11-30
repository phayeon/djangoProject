from django.db import models

from blog.posts.models import Post


class Tag(models.Model):
    use_in_migration = True
    tag_id = models.AutoField(primary_key=True)
    title = models.TextField()

    post = models.ForeignKey(Post, on_delete=models.CASCADE)

    class Meta:
        db_table = "blog_tag"

    def __str__(self):
        return f'{self.pk} {self.title}'
