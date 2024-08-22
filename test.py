# There is an array of  integers. There are also  disjoint sets,  and , each containing  integers. You like all the integers in set  and dislike all the integers in set . Your initial happiness is . For each  integer in the array, if , you add  to your happiness. If , you add  to your happiness. Otherwise, your happiness does not change. Output your final happiness at the end.

# Note: Since  and  are sets, they have no repeated elements. However, the array might contain duplicate elements.

# Constraints



# Input Format

# The first line contains integers  and  separated by a space.
# The second line contains  integers, the elements of the array.
# The third and fourth lines contain  integers,  and , respectively.

# Output Format

# Output a single integer, your total happiness.

# Sample Input

# 3 2
# 1 5 3
# 3 1
# 5 7
# Sample Output

# 1
# Explanation

# You gain  unit of happiness for elements  and  in set . You lose  unit for  in set . The element  in set  does not exist in the array so it is not included in the calculation.

# Hence, the total happiness is .
# ChatGPT said:
# ChatGPT
# To solve this problem, the approach involves iterating through each element of the array and checking if the element belongs to either of the given sets 
# 𝐴
# A or 
# 𝐵
# B. Based on the membership, we update the happiness score accordingly.

# Steps to Solve:
# Input Parsing:

# Read the integers 
# 𝑛
# n (length of the array) and 
# 𝑚
# m (size of the sets 
# 𝐴
# A and 
# 𝐵
# B).
# Read the array of 
# 𝑛
# n integers.
# Read the elements of set 
# 𝐴
# A (integers that you like).
# Read the elements of set 
# 𝐵
# B (integers that you dislike).
# Calculate Happiness:

# Initialize the happiness score to 0.
# Iterate over each element in the array:
# If the element is in set 
# 𝐴
# A, increase the happiness by 1.
# If the element is in set 
# 𝐵
# B, decrease the happiness by 1.
# If the element is in neither set, happiness remains unchanged.
# Output:

# Print the final happiness score.
# Python Implementation:
# python
# Copy code
# # Input Parsing
# n, m = map(int, input().split())  # n is the length of the array, m is the size of sets A and B
# array = list(map(int, input().split()))  # The array of n integers
# A = set(map(int, input().split()))  # Set A containing m integers you like
# B = set(map(int, input().split()))  # Set B containing m integers you dislike

# # Initialize happiness
# happiness = 0

# # Calculate happiness
# for num in array:
#     if num in A:
#         happiness += 1
#     elif num in B:
#         happiness -= 1

# # Output the final happiness
# print(happiness)
# Explanation:
# Input:

# 𝑛
# n and 
# 𝑚
# m represent the number of elements in the array and the size of sets 
# 𝐴
# A and 
# 𝐵
# B.
# The array contains 
# 𝑛
# n integers.
# 𝐴
# A is the set of integers you like.
# 𝐵
# B is the set of integers you dislike.
# Logic:

# Iterate over each integer in the array. Check if it belongs to set 
# 𝐴
# A (increment happiness) or set 
# 𝐵
# B (decrement happiness).