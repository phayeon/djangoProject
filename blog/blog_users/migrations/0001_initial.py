# Generated by Django 4.1.3 on 2022-12-07 03:58

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Blog_users',
            fields=[
                ('bu_id', models.AutoField(primary_key=True, serialize=False)),
                ('email', models.TextField()),
                ('nickname', models.TextField()),
                ('password', models.TextField()),
            ],
            options={
                'db_table': 'blog_user',
            },
        ),
    ]
