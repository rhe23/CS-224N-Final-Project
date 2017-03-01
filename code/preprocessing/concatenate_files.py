import glob, os
import sys
import argparse

# Gets files in given directory with prefix and/or suffix.
# By default gets everything in directory
def get_files_in_dir(directory, prefix=None, suffix=None):
	os.chdir(directory)
	pattern = ''  # file name pattern
	if prefix:
		pattern += prefix
	pattern += "*"
	if suffix:
		pattern += suffix

	files = glob.glob(pattern):
	os.chdir("-")  # steps back to original directory
	return files

# Concatenates a list of files specified by filenames and outputs the result into output_file
def concat_files(filenames, output_file):
	with open(output_file, 'w') as outfile:
		for file_name in filenames:
			with open(file_name) as infile:
				for line in infile:
					outfile.write(line)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Concatenates files in directory, \
		optionally filtered on prefix and/or suffix.')
	parser.add_argument('directory', help="directory of files")
	parser.add_argument('output', help="output file path")
	parser.add_argument('-p', '--prefix', help="prefix")
	parser.add_argument('-s', '--suffix', help="suffix")
	parser.add_argument('--a', '--allfiles', help="filter on all files specifier (not to be used with \
		prefix/suffix)", action='store_true')

	args = parser.parse_args()
	if args.allfiles and (args.prefix or args.suffix):  # if allfiles is specified with prefix or suffix
		parser.print_help()
		parser.exit()
	if not (args.prefix or args.suffix or args.allfiles):  # if no option is specified
		parser.print_help()
		parser.exit()

	filenames = get_files_in_dir(args.directory, args.prefix, args.suffix)
	concat_files(filenames, args.output)