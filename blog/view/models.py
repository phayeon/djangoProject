from django.db import models


class Views(models.Model):
    use_in_migration = True
    view_id = models.AutoField(primary_key=True)
    ip_address = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "blog_views"

    def __str__(self):
        return f'{self.pk} {self.ip_address} {self.created_at}'
