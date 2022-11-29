from django.db import models


class Hello(models.Model):
    use_in_migrations = True
    hello = models.CharField(primary_key=True, max_length=30)
    name = models.TextField()

    class Meta:
        db_table = "hellos"

    def __str__(self):
        return f'{self.pk} {self.name}'
