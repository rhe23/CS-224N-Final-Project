#!/bin/bash

for subreddit in 'AskReddit' 'LifeProTips' 'nottheonion' 'news' 'science' 'trees' 'tifu' 'personalfinance' 'mildlyinteresting' 'interestingasfuck'

do
	python LSTM_model_single.py train -r "$subreddit" -lr 0.0004 -hs 100 -do 1

done


