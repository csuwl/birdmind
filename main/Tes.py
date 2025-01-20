import sys

from openai import OpenAI
import time


def add(*s):
    print(s)
    print(*s)


def plus(**s) -> str:
    print(s)
    print(*s)



class Dd:
    age=143
    def __init__(self):
        print("no params")

    def __iter__(self):
        pass

    def __next__(self):
        pass

    def __str__(self):
        return str(self.age)


if __name__ == '__main__':
    print(sys.path)
    add(1, 2, 3, "fdsa")
    plus(bs=1, a=2, c=3)
    api = OpenAI(api_key="")
    a = set('abracadabra')
    b = set('alacazam')

    s: list = [2, 3, 4, 5, 6, 7, 1, 98, 5, 32]
    ss = sorted(s)
    print(ss)
    print(chr(5))
    print(type(chr(5)))
    print([x ** 2 for x in s])
    print(Dd())
    print(Dd.age)

    sql = "INSERT INTO EMPLOYEE(FIRST_NAME, \
           LAST_NAME, AGE, SEX, INCOME) \
           VALUES ('%s', '%s',  %s,  '%s',  %s)" % \
          ('Mac', 'Mohan', 20, 'M', 2000)

    print(sql)
    print(time.daylight)
    localtime = time.localtime(time.time())
    print(localtime)
    print(f"{localtime.tm_year}-{localtime.tm_mon}-{localtime.tm_mday} {localtime.tm_hour}:{localtime.tm_min}:{localtime.tm_sec}")


    # print("a: ", a)
    # print("b: ", b)
    #
    # print("a-b: ", a - b)  # a 和 b 的差集
    #
    # print("a | b: ", a | b)  # a 和 b 的并集
    #
    # print("a & b: ", a & b)  # a 和 b 的交集
    #
    # print("a ^ b: ", a ^ b)  # a 和 b 中不同时存在的元素
