# import atexit
# import signal
import sys
# import curses


# class ConsoleManager:
#     _instance = None

#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super(cls).__new__(cls)
#             cls._instance._initialized = False
#         return cls._instance

#     def init(self):
#         if not self._initialized:
#             self.stdscr = curses.initscr()
#             self.current_line = 0

#             curses.curs_set(0)

#             # Set up signal handler for clean termination on interrupt
#             signal.signal(signal.SIGINT, self._handle_interrupt)

#             # Register a function to be called when the program exits
#             atexit.register(self.cleanup)

#             self._initialized = True

#     def print_line(self, text='', y=None, x=0):
#         self.init()
#         if y is None:
#             y = self.current_line
#         self.stdscr.addstr(y, x, text)
#         self.stdscr.refresh()
#         self.current_line += 1

#     def delete_line(self, y=None):
#         self.init()
#         if y is None:
#             y = self.current_line
#         self.stdscr.move(y, 0)
#         self.stdscr.clrtoeol()
#         self.stdscr.refresh()

#     def inc(self):
#         self.init()
#         self.current_line += 1

#     def print_progress_bar(self, iter: int, total, y=None, prefix='', suffix='', length=35, fill='█', shown: str = 'num'):
#         self.init()
#         if y is None:
#             y = self.current_line
#         percent = iter / float(total) * 100
#         filled_length = int(length * percent / 100)
#         bar = fill * filled_length + '-' * (length - filled_length)
#         num = f"{iter}/{int(total)}" if shown == 'num' else f"{percent:.1f}%"
#         self.stdscr.addstr(
#             y, 0, f"{prefix} [{bar}] {num} {suffix}")
#         self.stdscr.refresh()

#     def _handle_interrupt(self, signum, frame):
#         # Exit the program without clearing the screen
#         sys.exit(0)

#     def cleanup(self):
#         # Leave alternative screen mode before cleanup
#         curses.curs_set(2)
#         curses.endwin()

def clear_line(length: int = 100):
    sys.stdout.write('\r' + ' ' * length)
    sys.stdout.write('\r')
    sys.stdout.flush()


def print_spinner(i: int, scale: int = 10_000):
    spinners = ['|', '/', '-', '\\']
    i = int(i / scale)
    print(f'\r{spinners[i % 4]}', end='', flush=True)


def print_progress_bar(iter: int, total, prefix='', suffix='', length=35, fill='█', shown: str = 'num'):
    percent = ("{0:.1f}%").format(100 * (iter / float(total)))
    filled_length = int(length * iter // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    num = f"{iter}/{int(total)}" if shown == 'num' else percent
    clear_line()
    sys.stdout.write(f"{prefix} [{bar}] {num} {suffix}")
    sys.stdout.flush()
