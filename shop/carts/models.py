from django.db import models

from shop.products.models import Product
from shop.shop_users.models import Shop_user


class Cart(models.Model):
    use_in_migration = True
    cart_id = models.AutoField(primary_key=True)

    shop_user = models.ForeignKey(Shop_user, on_delete=models.CASCADE)
    products = models.ForeignKey(Product, on_delete=models.CASCADE)

    class Meta:
        db_table = "shop_cart"

    def __str__(self):
        return f'{self.pk}'
