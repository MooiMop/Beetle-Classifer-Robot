import time
import datetime


def logprint(string, color='reset', timeout=0):
    colors = {
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'blue': '\033[34m',
        'reset': '\033[0m',
    }

    current_time = time.strftime('%H:%M', time.localtime())
    print(colors[color] + f'{current_time}    {str(string)}' + colors['reset'])
    time.sleep(timeout)


def print_dict(dict):
    print('\n')
    for key in dict:
        print(f'{key}:  {dict[key]}')
    print('\n')


TIME_FORMATS = {
    'date': '%Y.%m.%d',
    'time': '%H:%M',
    'datetime': '%Y.%m.%d %H:%M',
}


def get_time(mode='str', format='datetime'):
    time = datetime.datetime.now()
    if mode == 'datetime':
        return time
    elif mode == 'str':
        try:
            return time.strftime(TIME_FORMATS[format])
        except KeyError:
            return time.strftime(format)


def convert_time(time, format='datetime'):
    if type(time) is datetime.datetime:
        try:
            return time.strftime(TIME_FORMATS[format])
        except KeyError:
            return time.strftime(format)
    elif type(time) is str:
        try:
            return datetime.datetime.strptime(TIME_FORMATS[format])
        except KeyError:
            return datetime.datetime.strptime(format)


class bcolors:

    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[34m'
    RESET = '\033[0m'

    def green(string):
        return(bcolors.GREEN + string + bcolors.RESET)

    def red(string):
        return(bcolors.RED + string + bcolors.RESET)

    def yellow(string):
        return(bcolors.YELLOW + string + bcolors.RESET)

    def blue(string):
        return(bcolors.BLUE + string + bcolors.RESET)
