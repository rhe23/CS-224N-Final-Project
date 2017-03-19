#!/bin/bash

for subreddit in 'news' 'science' 'trees' 'tifu' 'personalfinance' 'mildlyinteresting' 'interestingasfuck'

do
	echo "$subreddit"
	python LSTM_model_single.py train -r "$subreddit" -lr 0.0004 -hs 100 -do 1

done


