from src.euphemia.orders import *
from src.euphemia.order_books import *
from src.euphemia.solvers import *
from src.euphemia.ploters import *

# Same price but different Order Books
OB1 = SimpleOrderBook([
    LinearOrder("Supply", 5, 15, 10),
    LinearOrder("Demand", 15, 5, 10)
])
OB2 = SimpleOrderBook([
    LinearOrder("Supply", 0, 10, 20),
    LinearOrder("Demand", 12, 10, 20)
])
OB3 = SimpleOrderBook([
    LinearOrder("Supply", 9, 11, 10),
    LinearOrder("Demand", 15, 0, 15)
])

ploter = ExamplePloter([OB1, OB2, OB3])
ploter.display()
