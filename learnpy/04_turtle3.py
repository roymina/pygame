import turtle

t = turtle.Turtle()
t.speed(0)
turtle.bgcolor("black")
t.pensize(2)

colors = ["cyan", "magenta", "yellow", "white"]
for i in range(36):
    t.color(colors[i % 4])
    t.circle(100)
    t.left(10)

turtle.done()