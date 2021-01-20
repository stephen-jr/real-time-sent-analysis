def delay():
    for a in range(1, 110000000):
        pass


i = 1
while True:
    print(i)
    i += 2
    delay()
    if i == 11:
        raise TypeError('Generic error')
