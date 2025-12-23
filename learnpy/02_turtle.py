import turtle

t = turtle.Turtle()
colors = ['red', 'purple', 'blue', 'green', 'orange', 'yellow']
t.speed(0)

for x in range(100):
    t.pencolor(colors[x % 6])
    t.width(x / 100 + 1)
    t.forward(x)
    t.left(59)

turtle.done()