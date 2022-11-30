from django.db import models


class Comment(models.Model):
    use_in_migration = True
    comment_id = models.AutoField(primary_key=True)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "blog_comment"

    def __str__(self):
        return f'{self.pk} {self.content} {self.created_at} {self.updated_at}'
