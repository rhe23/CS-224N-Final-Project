import glob, os
import sys
import argparse

# Gets files in given directory with prefix and/or suffix.
# By default gets everything in directory
def get_files_in_dir(directory, prefix=None, suffix=None):
	cwd = os.getcwd()
	os.chdir(directory)
	pattern = ''  # file name pattern
	if prefix:
		pattern += prefix
	pattern += "*"
	if suffix:
		pattern += suffix

	files = glob.glob(pattern)
	os.chdir(cwd)  # steps back to original directory
	return files

# Concatenates a list of files specified by filenames and outputs the result into output_file
def concat_files(directory, files, output_file, labels=None):
	labels_delimiter = '|||||'
	with open(output_file, 'w') as outfile:
		for file_name in files:
			subreddit = file_name.split("_")[0]  # Characters before first underscore
			file_path = directory + file_name
			with open(file_path) as infile:
				for line in infile:
					if labels:
						outfile.write(subreddit + ' ' + labels_delimiter + ' ')
					outfile.write(line)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Concatenates files in directory, \
		filtered on prefix and/or suffix.')
	parser.add_argument('directory', help="directory of files")
	parser.add_argument('output', help="output file path")
	parser.add_argument('-p', '--prefix', help="prefix")  # typically subreddits
	parser.add_argument('-s', '--suffix', help="suffix")
	parser.add_argument('-l', '--labels', help="add subreddit labels to each line of dataset", action='store_true')

	args = parser.parse_args()

	if not (args.prefix or args.suffix):  # if no option is specified
		print "Error: must specify at least one of prefixes/suffixes."
		parser.print_help()
		parser.exit()

	files = get_files_in_dir(args.directory, args.prefix, args.suffix)
	concat_files(args.directory, files, args.output, args.labels)
	print "Successfully concatenated {} files".format(len(files))