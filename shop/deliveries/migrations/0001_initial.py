# Generated by Django 4.1.3 on 2022-11-30 03:46

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Deliveries',
            fields=[
                ('delivery_id', models.AutoField(primary_key=True, serialize=False)),
                ('username', models.TextField()),
                ('address', models.TextField()),
                ('detail_address', models.TextField()),
                ('phone', models.TextField()),
            ],
            options={
                'db_table': 'shop_deliveries',
            },
        ),
    ]
