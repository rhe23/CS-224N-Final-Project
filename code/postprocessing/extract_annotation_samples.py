import csv
import argparse
import random

def get_all_titles(file):
	all_titles = {}
	f = open(args.posts_file, 'r')
	reader = csv.reader(f, delimiter=',')
	for row in reader:
		subreddit = row[0]
		title = row[1]
		if subreddit in all_titles:
			all_titles[subreddit].append(title)
		else:
			all_titles[subreddit] = [title]
	f.close()
	return all_titles

def get_sample_titles(all_titles, num_posts_per_sub):
	sample_titles = {}
	for subreddit, titles in all_titles.iteritems():
		sample_titles[subreddit] = random.sample(titles, num_posts_per_sub)
	return sample_titles

def generate_annotation_file(output_file, sample_titles, num_posts):
	f = open(output_file, 'w')
	writer = csv.writer(f, delimiter=',')
	for subreddit, titles in sample_titles.iteritems():
		for title in titles:
			writer.writerow([subreddit, title])
	f.close()

if __name__ == "__main__":
	random.setseed(1)
	parser = argparse.ArgumentParser(description="Extract a number of samples to be annotated")
	parser.add_argument('posts_file', help="file path for generated post titles")
	parser.add_argument('output_file', help="file path for outputting the annotation file")
	parser.add_argument('num_posts', help="number of posts to be generated PER SUBREDDIT")

	args = parser.parse_args()
	posts_file = args.posts_file
	output_file = args.output_file
	num_posts = int(args.num_posts)

	all_titles = get_all_titles(posts_file)
	sample_titles = get_sample_titles(all_titles, num_posts)
	generate_annotation_file(output_file, sample_titles, num_posts)
