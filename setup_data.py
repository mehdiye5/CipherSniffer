import os
import random

from sniffer.ciphers import substitution, transposition, reverse, shift, wordflip

random.seed(0)

def clean_gigaword_data(INPUT_DATA_PATHS, CLEANED_DATA_PATHS):

	for infile, outfile in zip(INPUT_DATA_PATHS, CLEANED_DATA_PATHS):

		if os.path.exists(outfile):
			print("Already Exists - \'{}\'".format(outfile))
			continue

		with open(outfile, "w") as f:

			# Only keeping words with alpha characters
			for line in open(infile):
				res = []
				for word in line.split():
					add = True
					for char in word:
						if char.isalpha() == False:
							add = False
							break
					if add:
						res.append(word)

				if len(res) != 0:
					f.write(" ".join(res) + "\n")

def encrypt_data(CIPHER_DATA_PATH, CLEANED_DATA_PATHS):
	"""
	Creates encrypted training, validation, testing, and embedding data sets.
	"""
	
	# Check output directory exists
	if not os.path.exists(CIPHER_DATA_PATH):
		os.makedirs(CIPHER_DATA_PATH)

	# Check files dont already exist
	for fname in ["train.txt", "valid.txt", "test.txt"]:
		if os.path.exists(CIPHER_DATA_PATH + fname):
			print("Already Exists - \'{}\'".format(CIPHER_DATA_PATH + fname))
			return

	# Opening files to write to
	with open(CIPHER_DATA_PATH + 'train.txt', 'w') as ftrain, \
		 open(CIPHER_DATA_PATH + 'valid.txt', 'w') as fvalid, \
		 open(CIPHER_DATA_PATH + 'test.txt', 'w') as ftest, \
		 open(CIPHER_DATA_PATH + 'embedding.txt', 'w') as fembed:

		# Counter
		c = 0

		# Opening files to read from
		for fpath in CLEANED_DATA_PATHS:
			with open(fpath, "r") as readfile:
				# Keep reading until at end of each file
				while True:
					line = readfile.readline()
					if not line:
						break
					
					# Encrypt Text
					encrypted_text = sniffer_encryption(c, line.rstrip())
					if not encrypted_text.endswith('\n'):
						encrypted_text += "\n"

					# Write to file
					if c < 50000:
						ftrain.write(encrypted_text)
					elif c < 52500:
						fvalid.write(encrypted_text)
					elif c < 55000:
						ftest.write(encrypted_text)
					else:
						# We dont need the label for embedding training
						fembed.write(encrypted_text[2:].strip() + " ")
					c+=1

def sniffer_encryption(i, text):
	"""
	Create encryption based on ith index. This guarantees and equal distribution
	across classes.
	"""
	i%=6
	if i == 0:
		return str(i) + " " + substitution(text)
	elif i == 1:
		return str(i) + " " + transposition(text)
	elif i == 2:
		return str(i) + " " + reverse(text)
	elif i == 3:
		return str(i) + " " + shift(text)
	elif i == 4:
		return str(i) + " " + wordflip(text)
	elif i == 5:
		return str(i) + " " + text # uncorrupted text

def main():

	# Clean gigaword data
	INPUT_DATA_PATHS = ["./data/original/train.article.txt","./data/original/valid.article.filter.txt"]
	CLEANED_DATA_PATHS = ["./data/original/clean.train.article.txt","./data/original/clean.valid.article.filter.txt"]
	# clean_gigaword_data(INPUT_DATA_PATHS, CLEANED_DATA_PATHS)

	# Create encrypted train and validation sets
	CIPHER_DATA_PATH = "./data/cipherdata/"
	encrypt_data(CIPHER_DATA_PATH, CLEANED_DATA_PATHS)
	
if __name__ == '__main__':
	main()