import salvo.run


args = {
    'method': 'GET',
    'concurrency': 4,
    'requests': 1000
}
print(f'1 args={args}')
salvo.run.load(url='https://www.youtube.com/', args=args)
