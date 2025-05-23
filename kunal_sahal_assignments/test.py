def right_aligned(height, symbol):
    """To print right aligned triangle"""

    for n in range(height):
        print()
        for s in range(height, n, -1):
            print(" ", end="")
        for m in range(n + 1):
            print(symbol, end="")
    return -1


def equilateral(height, symbol):
    """To print right aligned triangle"""

    a = 1
    for n in range(height):
        print()
        for s in range(height, n, -1):
            print(" ", end="")
        for m in range(a):
            print(symbol, end="")
        a += 2


def diamond(height, symbol):
    """To print diamond shape"""

    # pyramid portion
    a = 1
    for n in range(height):
        print()
        for s in range(height, n, -1):
            print(" ", end="")
        for m in range(a):
            print(symbol, end="")
        a += 2
    # inverse pyramid portion
    a -= 4
    for n in range(1, height):
        print()
        for s in range(-1, n):
            print(" ", end="")
        for m in range(a):
            print(symbol, end="")
        a -= 2


try:
    var = int(
        input(
            "1. For Right Alinged Triangle \n2. For Equilateral Triangle \n3. For Diamond \nYour Choice :"
        )
    )
    # to check for non-numeric values
    height = int(input("Enter height of the triangle : "))
    # to check for negative integer values
    if height < 0:
        raise ValueError
except ValueError:
    print("Invalid input")
else:
    symbol = str(input("Character to be print in the form of triangle : "))

match var:
    case 1:
        print("insdie right_alinged function")
        right_aligned(height, symbol)
    case 2:
        equilateral(height, symbol)
    case 3:
        diamond(height, symbol)
    case _:
        print('invalid input')
