from django.db import models


class Showtime(models.Model):
    use_in_migration = True
    showtime_id = models.AutoField(primary_key=True)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()

    class Meta:
        db_table = "movie_showtime"

    def __str__(self):
        return f'{self.pk} {self.start_time} {self.end_time}'
