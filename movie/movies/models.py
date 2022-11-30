from django.db import models


class Movies(models.Model):
    use_in_migration = True
    movie_id = models.AutoField(primary_key=True)
    title = models.TextField()
    director = models.TextField()
    description = models.TextField()
    poster_url = models.TextField()
    running_time = models.TextField()
    age_rating = models.TextField()

    class Meta:
        db_table = "movie_movies"

    def __str__(self):
        return f'{self.pk} {self.title} {self.director} {self.description}' \
               f' {self.poster_url} {self.running_time} {self.age_rating}'
