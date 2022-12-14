# Generated by Django 4.1.3 on 2022-12-07 03:58

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('shop_users', '0001_initial'),
        ('products', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Cart',
            fields=[
                ('cart_id', models.AutoField(primary_key=True, serialize=False)),
                ('products', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='products.product')),
                ('shop_user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='shop_users.shop_user')),
            ],
            options={
                'db_table': 'shop_cart',
            },
        ),
    ]
