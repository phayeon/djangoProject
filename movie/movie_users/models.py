from django.db import models


class Movie_user(models.Model):
    use_in_migration = True
    mu_id = models.AutoField(primary_key=True)
    email = models.TextField()
    nickname = models.TextField()
    password = models.TextField()
    age = models.TextField()

    class Meta:
        db_table = "movie_user"

    def __str__(self):
        return f'{self.pk} {self.email} {self.nickname} {self.password} {self.age}'

