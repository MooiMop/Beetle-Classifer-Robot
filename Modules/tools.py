import time

def logprint(string):
    current_time = time.strftime('%H:%M',time.localtime())
    print(f'{current_time}  {str(string)}')
    time.sleep(0.5)
