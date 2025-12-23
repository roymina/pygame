import random

number = random.randint(1, 100)
print("我想了一个 1 到 100 之间的数字，你能猜中吗？")

while True:
    guess = int(input("你的猜测："))
    if guess < number:
        print("太小了！")
    elif guess > number:
        print("太大了！")
    else:
        print("🎉 恭喜你，猜对了！")
        break