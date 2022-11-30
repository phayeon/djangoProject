from django.db import models
from movie.movie_users.models import Movie_user
from movie.showtimes.models import Showtime
from movie.theaters.models import Theater


class TheaterTicket(models.Model):
    use_in_migration = True
    ticket_id = models.AutoField(primary_key=True)
    x = models.IntegerField()
    y = models.IntegerField()

    movie_user = models.ForeignKey(Movie_user, on_delete=models.CASCADE)
    showtime = models.ForeignKey(Showtime, on_delete=models.CASCADE)
    theater = models.ForeignKey(Theater, on_delete=models.CASCADE)

    class Meta:
        db_table = "movie_theater_ticket"

    def __str__(self):
        return f'{self.pk} {self.x} {self.y}'
