#!/bin/bash

for subreddit in 'AskReddit' 'LifeProTips' 'nottheonion' 'news' 'science' 'trees' 'tifu' 'personalfinance' 'mildlyinteresting' 'interestingasfuck'

do
	echo "$subreddit"
	python LSTM_model_variable_lengths.py generate -g "$subreddit" -nw 15 -n 100 -l 1 -sq 2 -hs 200 -p False

done

