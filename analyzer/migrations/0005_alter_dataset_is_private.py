# Generated by Django 3.2.23 on 2025-04-24 18:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analyzer', '0004_auto_20250424_2150'),
    ]

    operations = [
        migrations.AlterField(
            model_name='dataset',
            name='is_private',
            field=models.BooleanField(default=True),
        ),
    ]
