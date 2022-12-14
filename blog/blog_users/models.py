from django.db import models


# 유저
class BlogUser(models.Model):
    use_in_migration = True
    blog_userid = models.AutoField(primary_key=True)
    email = models.TextField()
    nickname = models.TextField()
    password = models.TextField()

    class Meta:
        db_table = "blog_users"

    def __str__(self):
        return f'{self.pk} {self.email} {self.nickname} {self.password}'
