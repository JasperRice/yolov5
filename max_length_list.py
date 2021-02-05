class Base(object):

    def __init__(self, size):
        self.size = size
        self.list = []

    def __len__(self):
        return len(self.list)

    def __str__(self):
        string = "Capacity: {}\nContent: {}".format(self.size, self.list)
        return string

    def empty(self):
        return not len(self)

    def full(self):
        return len(self) > self.size

    def push(self, obj):
        self.list.append(obj)
        while self.full():
            self.pop(0)

    def pop(self, index=0):
        return self.list.pop(index)
