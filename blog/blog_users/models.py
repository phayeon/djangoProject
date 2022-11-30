from django.db import models


class Blog_users(models.Model):
    use_in_migration = True
    bu_id = models.AutoField(primary_key=True)
    email = models.TextField()
    nickname = models.TextField()
    password = models.TextField()

    class Meta:
        db_table = "blog_user"

    def __str__(self):
        return f'{self.pk} {self.email} {self.nickname} {self.password}'
