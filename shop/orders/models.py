from django.db import models

from shop.deliveries.models import Deliveries
from shop.products.models import Product
from shop.shop_users.models import Shop_user


class Orders(models.Model):
    use_in_migration = True
    order_id = models.AutoField(primary_key=True)
    created_at = models.DateTimeField(auto_now_add=True)

    deliveries = models.ForeignKey(Deliveries, on_delete=models.CASCADE)
    shop_user = models.ForeignKey(Shop_user, on_delete=models.CASCADE)
    products = models.ForeignKey(Product, on_delete=models.CASCADE)

    class Meta:
        db_table = "shop_orders"

    def __str__(self):
        return f'{self.pk} {self.created_at}'
