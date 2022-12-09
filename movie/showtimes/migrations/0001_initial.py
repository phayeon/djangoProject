# Generated by Django 4.1.3 on 2022-12-07 03:58

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('theaters', '0001_initial'),
        ('movies', '0001_initial'),
        ('cienmas', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Showtime',
            fields=[
                ('showtime_id', models.AutoField(primary_key=True, serialize=False)),
                ('start_time', models.DateTimeField()),
                ('end_time', models.DateTimeField()),
                ('cinema', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='cienmas.cinema')),
                ('movie', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='movies.movies')),
                ('theater', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='theaters.theater')),
            ],
            options={
                'db_table': 'movie_showtime',
            },
        ),
    ]
