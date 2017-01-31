#!/usr/bin/env python3

class LookAndSay:
    def __init__(self, level):
        self.level = level
        self.n = -1
        self.count = 0
        if(level > 1):
            self.under_las = LookAndSay(level - 1)
            self.clist = []
        else:
            self.clist = [1]

    def get(self):
        if(len(self.clist) != 0):
            return self.clist.pop()

        if(self.level == 1):
            return -1

        while True:
            an = self.under_las.get()
            if(self.n == -1):
                if(an == -1):
                    return -1
                else:
                    self.n = an
                    self.count = 0
            
            if(an != self.n):
                self.clist.insert(0, self.n)
                self.clist.insert(0, self.count)
                self.n = an
                self.count = 1
                break
            else:
                self.count += 1

        if(len(self.clist) == 0):
            return -1

        return self.clist.pop()

