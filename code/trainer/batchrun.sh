#!/bin/bash

for subreddit in 'AskReddit' 'LifeProTips' 'nottheonion' 'news' 'science' 'trees' 'tifu' 'personalfinance' 'mildlyinteresting' 'interestingasfuck'

do
	echo "$subreddit"
	python LSTM_model_variable_lengths.py train -r "$subreddit" -lr 0.0004 -hs 200 -do 1 -l 1 -sq 3 -p 0

done

