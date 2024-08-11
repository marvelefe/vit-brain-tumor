
import arrow


startTime = arrow.now().timestamp()


print(startTime)


while True:
    print(arrow.get(arrow.now().timestamp() - startTime).format('HH:mm:ss'))


    