import pandas as pd
import random
from datetime import datetime, timedelta

# Function to generate random Russian names
def generate_russian_names(count):
    first_names = [
        'Александр', 'Дмитрий', 'Максим', 'Сергей', 'Андрей', 'Алексей', 'Артём', 'Илья', 'Кирилл', 'Михаил',
        'Никита', 'Матвей', 'Роман', 'Егор', 'Арсений', 'Иван', 'Денис', 'Евгений', 'Тимофей', 'Владислав',
        'Анастасия', 'Мария', 'Дарья', 'Анна', 'Алёна', 'Ирина', 'Екатерина', 'Татьяна', 'Софья', 'Ольга'
    ]

    last_names = [
        'Иванов', 'Смирнов', 'Кузнецов', 'Попов', 'Васильев', 'Петров', 'Соколов', 'Михайлов', 'Новиков', 'Фёдоров',
        'Морозов', 'Волков', 'Алексеев', 'Лебедев', 'Семёнов', 'Егоров', 'Павлов', 'Козлов', 'Степанов', 'Николаев'
    ]

    names = []
    for i in range(count):
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        names.append(f"{first_name} {last_name}")
    return names

# Function to generate random payment data
def generate_payment_data(count):
    names = generate_russian_names(count)

    # Headers
    data = []
    data.append(['ID', 'ФИО', 'Дата платежа', 'Сумма (руб.)', 'Тип платежа', 'Статус'])

    payment_types = ['Кредит', 'Аренда', 'Коммунальные услуги', 'Интернет', 'Телефония', 'Страхование', 'Образование', 'Медицина', 'Наркотики', 'Проститутки', 'Еда', 'Токены']
    statuses = ['Оплачено', 'Задолженность', 'Частично оплачено', 'Телеметрия', 'Частично отпижен']

    for i in range(1, count + 1):
        id = i
        name = names[i-1]

        # Generate random date in 2023-2024
        year = random.choice([2023, 2024])
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        date = f"{day:02d}.{month:02d}.{year}"

        # Generate random amount between 1,000 and 100,000 rubles
        amount = random.randint(1000, 100000)

        payment_type = random.choice(payment_types)
        status = random.choice(statuses)

        data.append([id, name, date, amount, payment_type, status])

    return data

# Generate 100 records
payment_data = generate_payment_data(100)

# Create DataFrame
df = pd.DataFrame(payment_data[1:], columns=payment_data[0])

# Save to Excel file
df.to_excel('russian_monthly_payments_python.xlsx', index=False, sheet_name='Платежи')

print('Excel file "russian_monthly_payments_python.xlsx" with 100 records has been created successfully!')