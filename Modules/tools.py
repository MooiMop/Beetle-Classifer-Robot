import time


def logprint(string, color='reset', timeout=0.5):
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
