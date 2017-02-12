#!/bin/bash

# Compiles a teX and bib files into pdf's
# You can run this command from the directory that contains the teX and bib file
# Alternatively, you can specify an extra directory path argument when running
# This command will compile all teX and corresponding bib files in the directory 

DIR_PATH="."
if [ $# -gt 1 ]; then
    echo "Usage: ./compile_latex [optional dir path]"
    echo "If no dir path specified, current dir path is assumed"
    exit 1
elif [ $# -eq 1 ]; then
    DIR_PATH=$1
    if [ ! -d $DIR_PATH ]; then
	echo "Error: Specified directory does not exist"
	echo "Usage: ./compile_latex [optional dir path]"
	exit 1
    fi
fi

cd $DIR_PATH
# Find all file names in DIR_PATH with extension ".tex"
file_names=($(find . -maxdepth 1 -type f -name "*.tex"))
if [ ${#file_names} -eq 0 ]; then
    echo "No tex files found in directory."
fi
for tex_file in $file_names;
do
    base_file=${tex_file::${#tex_file}-4}  # tex_file with extension ".tex" removed
    bib_file="$base_file.bib" 
    if [ ! -f $bib_file ]; then
	echo "bib file not found for $base_file"
    else
	pdflatex $base_file
	biber $base_file
	pdflatex $base_file
	pdflatex $base_file
    fi
done

cd -