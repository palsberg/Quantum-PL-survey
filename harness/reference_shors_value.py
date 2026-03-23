from typing import List
# def calculate_shors_factors(n: int) -> List[int]:
#     """return the list of all possible prime factors of n"""
#     if n < 2:
#         raise ValueError("Input must be an integer greater than or equal to 2.")
    
#     prime_factors = [] # List to store the distinct prime factors of n
#     cur_factor = 2
#     while cur_factor <= n:
#         if n % cur_factor == 0:
#             prime_factors.append(cur_factor)

#         while n % cur_factor == 0:
#             n //= cur_factor
#         cur_factor += 1
    
#     return prime_factors

def calculate_shors_factors(n: int) -> List[int]:
    """return the list of all possible prime factors of n"""
    if n < 2:
        raise ValueError("Input must be an integer greater than or equal to 2.")
    if n == 2:
        return [2]
    
    factors = [] # List to store the distinct prime factors of n
    cur_factor = 2
    while cur_factor <= n-1:
        if n % cur_factor == 0:
            factors.append(cur_factor)

        cur_factor += 1
    
    return factors

def test_shors_value(factors: List[int], shors_value: int) -> bool:
    """given a list of factors and a Shor's value, return True if the factors are correct."""
    if shors_value in factors:
        return True
    return False

def main():
    n = 15
    factors = calculate_shors_factors(n)
    print(f"Factors of {n}: {factors}")
    
    shors_value = 3
    if test_shors_value(factors, shors_value):
        print(f"{shors_value} is a correct factor of {n}.")
    else:
        print(f"{shors_value} is not a correct factor of {n}.")

if __name__ == "__main__":
    main()