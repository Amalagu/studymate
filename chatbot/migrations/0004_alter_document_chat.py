# Generated by Django 5.0.1 on 2024-02-15 05:58

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("chatbot", "0003_document"),
    ]

    operations = [
        migrations.AlterField(
            model_name="document",
            name="chat",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="documents",
                to="chatbot.chatmodel",
            ),
        ),
    ]
