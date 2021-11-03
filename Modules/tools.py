import time


def logprint(string):

    current_time = time.strftime('%H:%M', time.localtime())
    print(f'{current_time}  {str(string)}')
    time.sleep(0.5)


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
