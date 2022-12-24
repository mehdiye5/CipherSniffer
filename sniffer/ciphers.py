import string
import random


def caesar(text, shift=None):
	"""
	Applies a Caesar Cipher to a body of text.
	
	Unique Encryptions: 25
	More Info: https://ciphertools.co.uk/InfoCaesar.php
	"""

	# Validates shift input
	if shift == None:
		shift = random.randint(1,25)
	elif shift <= 0 or shift%26==0:
		raise ValueError('Shift must be >0 and not divisible 26 ')
	shift%=26

	# Apply cipher
	mapping = string.ascii_lowercase[shift:] + string.ascii_lowercase[:shift]
	text = text.translate(str.maketrans(string.ascii_lowercase, mapping))
	
	return text

def affine(text, multiplier=None, shift=None):
	"""
	Applies an Affine Cipher to a body of text. In this implementation, we do not
	include the 25 caeser ciphers as well.
	
	Unique Encryptions: 286
	More Info: https://ciphertools.co.uk/InfoAffine.php
	"""

	# Validate multiplier
	multiplier_options = [3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25] # values that generate unique encryptions
	if multiplier == None:
		multiplier = random.choice(multiplier_options)
	elif multiplier not in multiplier_options:
		raise ValueError("Invalid Multiplier")

	# Validate shift
	if shift == None:
		shift = random.randint(0,25)
	elif shift < 0 or (shift > 0 and shift%26==0):
		raise ValueError('Shift must be >= 0 and not divisible 26 ')
	shift%=26

	# Create Mapping
	mapping = ""
	for i in range(26):
		mapping += chr((((i*multiplier)+shift)%26) + ord('a'))

	# Apply cipher
	text = text.translate(str.maketrans(string.ascii_lowercase, mapping))	
	return text

def substitution(text, der=True):
	"""
	A substitution cipher takes the normal alphabet and substitutes each 
	letter for another letter from the alphabet.

	Here we also add a derangement condition. This means that no letter can 
	map to itself.

	Unique Encryptions (w/ derangement): 148,362,637,348,470,135,821,287,825
	Unique Encryptions (w/out derangement): 403,291,461,126,605,635,584,000,000

	More Info: https://ciphertools.co.uk/InfoAffine.php
	Derangements: https://www.dcode.fr/derangements-generator
	"""
	while True:
		
		# Shuffle
		mapping = list(string.ascii_lowercase)
		random.shuffle(mapping)
		mapping = "".join(mapping)
		
		# Check derangement
		derangement = True
		for c1, c2 in zip(mapping, string.ascii_lowercase):
			if c1 == c2:
				derangement = False
				break

		if derangement == True:
			break

	text = text.translate(str.maketrans(string.ascii_lowercase, mapping))	
	return text

def transposition(text):
	"""
	Applies a transposition cipher to given text. Results in text that contains
	all the original characters but in a different order.
	
	Unique Encryptions: INF

	More Info: https://ciphertools.co.uk/InfoTransposition.php
	"""

	# Store word lengths
	res = ""
	lens = [len(y) for y in text.split()]

	# Shuffle letters
	text = list(text.replace(" ", ""))
	temp = text

	# Checks shuffle actually shuffles
	for i in range(100):
		random.shuffle(text)
		if text != temp:
			break

	# Reconstruct sentence
	res = ""
	i = 0
	for val in lens:
		res += "".join(text[i:i+val]) + " "
		i+=val
	res = res[:-1]
	
	return res

def reverse(text):
	"""
	Applies the simple reverse cipher to input text.
	
	Unique Encryptions: 1
	"""
	return text[::-1]

def shift(text):
	"""
	Applys a random to shift to all non-space characters. All
	positions of spaces are kept constant.

	Unique Encryptions: COUNT(alpha characters) - 1
	"""
	start = text
	iterations = 100

	# Loop to protect against rare edge case where shift results in original text
	for i in range(10):
		val = random.randint(1, len(start)-1)
		lens = [len(y) for y in start.split()]

		text = start[val:] + start[:val]
		p = 0
		res = ""

		for i, val in enumerate(lens):
			for j in range(val):
				while text[p] == " ":
					p+=1
				res += text[p]
				p+=1
			if i != len(lens)-1:
				res += " "

		# Checks if string equals original text (RARE EDGE CASE)
		if res == start:
			continue
		return res

	return None

def wordflip(text):
	"""
	Reverses every word in the input text, but retains
	the original order of words in the sentence:

	Unique Encryptions: 1
	"""
	return " ".join([word[::-1] for word in text.split()])