#!/bin/bash

for subreddit in 'AskReddit' 'LifeProTips' 'nottheonion' 'news' 'science' 'trees' 'tifu' 'personalfinance' 'mildlyinteresting' 'interestingasfuck'

do
	echo "$subreddit"
	python LSTM_model_variable_lengths.py generate -g "$subreddit" -nw 20 -n 100 -l 1 -sq 2 -hs 200 -p 0

done

