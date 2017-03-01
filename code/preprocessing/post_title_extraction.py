import glob, os
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

# extracts the post titles from all the raw filtered files and outputs them in separate files
if __name__ == "__main__":
	if len(sys.argv) != 2:
		print "Usage: python post_text_extraction.py <directory>"
	directory = sys.argv[1]

	cwd = os.getcwd()
	os.chdir(directory)
	for infile in glob.glob("*_raw"):
		outfile = infile[:-3] + "titles"
		extract_title_from_json(infile, outfile)
	os.chdir(cwd)