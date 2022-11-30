from django.db import models


class TheaterTicket(models.Model):
    use_in_migration = True
    ticket_id = models.AutoField(primary_key=True)
    x = models.IntegerField()
    y = models.IntegerField()

    class Meta:
        db_table = "movie_theater_ticket"

    def __str__(self):
        return f'{self.pk} {self.x} {self.y}'
