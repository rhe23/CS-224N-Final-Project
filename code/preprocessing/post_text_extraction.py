import json
import sys

def extract_title_from_json(input_file, output_file):
	fin = open(input_file, 'r')
	fout = open(output_file, 'w')
	for line in fin:
		post_json = json.loads(line)
		text = post_json['title']
		fout.write(text + "\n")
	fin.close()
	fout.close()

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print "Usage: python post_test_extraction.py <input-filename> <output-filename>"
	input_file = sys.argv[1]
	output_file = sys.argv[2]
	extract_title_from_json(input_file, output_file)