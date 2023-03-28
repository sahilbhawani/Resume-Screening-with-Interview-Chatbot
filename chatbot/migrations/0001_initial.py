# Generated by Django 4.1.2 on 2022-10-15 12:32

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Student',
            fields=[
                ('student_id', models.AutoField(primary_key=True, serialize=False)),
                ('student_name', models.CharField(default='', max_length=50)),
                ('student_email', models.CharField(default='', max_length=50)),
                ('student_college', models.CharField(default='', max_length=50)),
                ('student_cgpa', models.CharField(default='', max_length=50)),
                ('student_resume', models.FileField(default='', upload_to='media')),
                ('student_resume_result', models.CharField(default='', max_length=50)),
            ],
        ),
    ]
