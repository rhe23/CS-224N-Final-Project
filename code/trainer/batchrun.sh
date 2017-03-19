#!/bin/bash

for subreddit in 'AskReddit' 'LifeProTips' 'nottheonion' 'news' 'science' 'trees' 'tifu' 'personalfinance' 'mildlyinteresting' 'interestingasfuck'

do
	echo "$subreddit"
	python LSTM_model_single.py train -r "$subreddit" -lr 0.001 -hs 100 -do 1 -l 1

done


