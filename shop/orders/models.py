from django.db import models


class Orders(models.Model):
    use_in_migration = True
    order_id = models.AutoField(primary_key=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "shop_orders"

    def __str__(self):
        return f'{self.pk} {self.created_at}'
