# -*- coding: utf-8 -*-
"""Assign5_new.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XzI_EQlMRzOllt9v9Pl7xI26E9kjpcUG
"""

P10 = (3, 5, 2, 7, 4, 10, 1, 9, 8, 6)
P8 = (6, 3, 7, 4, 8, 5, 10, 9)
P4 = (2, 4, 3, 1)
IP = (2, 6, 3, 1, 4, 8, 5, 7)
IPi = (4, 1, 3, 5, 7, 2, 8, 6)
E = (4, 1, 2, 3, 2, 3, 4, 1)
S0 = [
 [1, 0, 3, 2],
 [3, 2, 1, 0],
 [0, 2, 1, 3],
 [3, 1, 3, 2]
 ]
S1 = [
 [0, 1, 2, 3],
 [2, 0, 1, 3],
 [3, 0, 1, 0],
 [2, 1, 0, 3]
 ]

def permutation(pattern, key):
 permuted = ""
 for i in pattern:
  permuted += key[i-1]
 return permuted

def generate_first(left, right):
 left = left[1:] + left[:1] # 5-bit left & right. Now last 4 bits selected & first bit added at last. result 5 bit only.
 right = right[1:] + right[:1]
 key = left + right #left & right added. result 10 bit key. But we need 8 bit key. so, permutate using P8.
 return permutation(P8, key)

def generate_second(left, right):
 left = left[3:] + left[:3]  # if done on LS-1 only LS-2. But now doing on left & right. So, shift by 3. 
 right = right[3:] + right[:3]
 key = left + right
 return permutation(P8, key) #But we need 8 bit key. so, permutate using P8.

def transform(right, key):
 extended = permutation(E, right)
 xor_cipher = bin(int(extended, 2) ^ int(key, 2))[2:].zfill(8)
 xor_left = xor_cipher[:4]
 xor_right = xor_cipher[4:]
 print("After Xor left",xor_left)
 print("After Xor right",xor_right)
 new_left = Sbox(xor_left, S0)
 new_right = Sbox(xor_right, S1)
 return permutation(P4, new_left[2:] + new_right[2:])

def Sbox(data, box):
 row = int(data[0] + data[3], 2)     #We take the first and fourth bit as row and the second and third bit as a column for our S boxes.
 column = int(data[1] + data[2], 2)
 return bin(box[row][column])[2:].zfill(4)

def encrypt(left, right, key): #int(string, Base). Converts into integer. Here binary numbers needed to convert into int. So, base 2 
 cipher = int(left, 2) ^ int(transform(right, key), 2)  
 return right, bin(cipher)[2:].zfill(4) #if length not 4 add zeros at start.

def decrypt(left, right, key): #int(string, Base). Converts into integer. Here binary numbers needed to convert into int. So, base 2 
 plain = int(left, 2) ^ int(transform(right, key), 2)  
 return right, bin(plain)[2:].zfill(4)

"""Algorithm"""

key = input("Enter a 10-bit key: ")
if len(key) != 10:
 raise Exception("Check the input")
plaintext = input("Enter 8-bit plaintext: ")
if len(plaintext) != 8:
 raise Exception("Check the input")

#key generation
p10key = permutation(P10, key)    #we need result 10 bit key, so permutate using P10.
print("First Permutation", p10key)
left_key = p10key[:len(p10key)//2] #why //2 --> to get middle element even in case of odd.
print("Left key",left_key)
right_key = p10key[len(p10key)//2:]
print("Right key",right_key)
first_key = generate_first(left_key, right_key)
print("*****")
print("First key:", first_key)
second_key = generate_second(left_key, right_key)
print("Second key", second_key)
print("*****")

#Encryption 
initial_permutation = permutation(IP, plaintext)
print("Initial Permutation",initial_permutation)
left_data = initial_permutation[:len(initial_permutation)//2]
right_data = initial_permutation[len(initial_permutation)//2:]
print("Left data",left_data)
left, right = encrypt(left_data, right_data, first_key)
left, right = encrypt(left,right, second_key)
ciphertext=permutation(IPi, right + left)
print("Ciphertext:", ciphertext )

initial_permutation = permutation(IP, ciphertext)
print("Initial Permutation",initial_permutation)
left_data = initial_permutation[:len(initial_permutation)//2]
right_data = initial_permutation[len(initial_permutation)//2:]
print("Left data",left_data)
left, right = decrypt(left_data, right_data, second_key)
print("After 1st round",left,right)
left, right = decrypt(left, right, first_key)
print(left,right)
newplaintext=permutation(IPi, right+left)
print("After decryption plaintext:", newplaintext )