"""
Author: Bipin P. (mailto: bipinp2013@gmail.com)
http://iambipin.com
101010101    10  101010101    10  101     10    101010101
1010101010   10  1010101010   10  1010    10    1010101010
10      101  10  10      101  10  10 01   10    10      101
10      101  10  10      101  10  10  10  10    10      101
1010101010   10  1010101010   10  10   01 10    1010101010
1010101010   10  101010101    10  10    1010    101010101
10      101  10  10           10  10     010    10
10      101  10  10           10  10      10    10
1010101010   10  10           10  10      10    10
101010101    10  10           10  10      10    10  10
Python Program to Convert Floating-Point Decimal to Binary number
"""
import numpy


def dec2bin(num, places):
    """"
    Function to convert a floating point decimal number to binary number
    """
    decList = []
    whole, dec = str(num).split('.')
    dec = int(dec)
    counter = 1
    whole = int(numpy.floor(num))
    wholeList = bin(whole).lstrip('0b')

    decproduct = dec
    while (counter <= places):
        decproduct = decproduct * (10 ** -(len(str(decproduct))))
        decproduct *= 2
        decwhole, decdec = str(decproduct).split('.')
        decwhole = int(decwhole)
        decdec = int(decdec)
        decList.append(decwhole)
        decproduct = decdec
        counter += 1

    return wholeList, ''.join(map(str, decList))
