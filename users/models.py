from django.db import models


class Users(models.Model):
    use_in_migrations = True
    users = models.CharField(primary_key=True, max_length=30)
    name = models.TextField()

    class Meta:
        db_table = "Users"

    def __str__(self):
        return f'{self.pk} {self.name}'
