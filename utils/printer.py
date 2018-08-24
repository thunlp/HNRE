import sys

class Printer(object):

    def __init__(self, ):

        super(Printer, self).__init__()

        self._maxlen = 0

    def __call__(self, x, inplace=True):
        
        if inplace:

            if len(x) >= self._maxlen:
                self._maxlen = len(x)
            else:
                x = x + ' ' * (self._maxlen - len(x))
            sys.stdout.write(x + '\r')
            sys.stdout.flush()

        else:

            self._maxlen = 0

            print(x)

        return self._maxlen

    @property

    def maxlen(self):
        
        return self._maxlen

    @maxlen.setter

    def maxlen(self, val):
        
        self._maxlen = val

        return val
