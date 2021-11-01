"""
Calculate the mu angle from given X and Y
"""

import numpy as np

R_Sun = 960

if __name__== "__main__":
    x = int(input("Enter x: "))
    y = int(input("Enter y: "))
    mu_angle = np.sqrt(R_Sun**2 - (x**2 + y**2))/R_Sun
    print(f"The mu_angle is {mu_angle:.2f}")
    print(f"The mu_angle^2 is {mu_angle**2:.2f}")
