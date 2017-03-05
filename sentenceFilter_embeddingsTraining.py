'''
Author: Tyler Chase
Date: 2017/03/02

This function takes in the address of a input json file, the address of an output 
json files, a list of subreddits you want filtered, 
and a boolean value as to whether you want the outputs printed to the console. 
It then returns a text file with the titles of the subreddits (one post per line)
''' 

# import libraries
import json 
import time


def import_data(input_address, output_address, subreddit = None, print_flag = False):
    if subreddit == None:
        raise NameError('Please enter a list of subreddits')
    # Open the file to be filtered
    f = open(input_address, "r")
    # Open the output file to be written to
    num_outputs_list = []
    f_o = open(output_address + 'top50Subreddits_sentences', "a") 
    
    # make list of posts read in per subreddit
    for i in subreddit:
        num_outputs_list.append(0)
               
    read_errors = 0
    ascci_errors = 0
    # Import JSON lines one by one
    posts_seen = 0
    for line in f:
        posts_seen+=1
        try:
            temp = json.loads(line)
            sub_iter = 0
            for sub in subreddit:
                if sub == temp['subreddit']:
                    # try to see if the title can be converted into ascci characters
                    try:
                        str(temp['title'])
                        # Form output, convert to json and write to output
                        output = temp['title']
                        f_o.write(output)
                        f_o.write('\n')
                        # If print_flag is true print to console
                        if print_flag:
                            print('subreddit:')
                            print(temp['subreddit'])
                            print('upvotes: ')
                            print(temp['ups'])
                            print('title:')
                            print(temp['title'])
                            print('\n')
                        num_outputs_list[sub_iter]+=1
                    except:
                        ascci_errors+=1
                sub_iter+=1
        except:
            read_errors+=1
    f.close()
    f_o.close()
                    
    # print the number of posts from each subreddit
    sub_iter = 0
    for i in subreddit:
        print(i + ' posts found: ' + str(num_outputs_list[sub_iter]) )
        sub_iter+=1        
           
    print('')    
    print('posts seen: ' + str(posts_seen))
    print('reading errors: ' + str(read_errors))
    print('ascci errors: ' + str(ascci_errors))   

    
# Test filter function
if __name__ == "__main__":
    
    main_start = start_time = time.time()
    
    # input and output addresses
    input_list = ['2015{:02d}/RS_2015-{:02d}'.format(i, i) for i in range(1,13)]
    
    input_address = "/Users/tylerchase/Documents/Stanford_Classes/CS224n_Natural_Language_Processing_with_Deep_Learning/final project/data/"
    output_address = "/Users/tylerchase/Documents/Stanford_Classes/CS224n_Natural_Language_Processing_with_Deep_Learning/final project/data/"
    
    # subreddit list
    # listed with similar subs on each line
    subreddits = ['AskReddit', 'LifeProTips',\
                  'nottheonion', 'news', 'science',\
                  'trees', \
                  'tifu', \
                  'personalfinance', \
                  'mildlyinteresting', 'interestingasfuck',\
                  'funny', 'todayilearned', 'worldnews', 'pics', 'IAmA',
                  'gaming', 'videos', 'movies', 'Music', 'aww', 'gifs', 
                  'explainlikeimfive', 'askscience', 'EarthPorn', 'books', 'television',
                  'DIY', 'Showerthoughts', 'space', 'sports', 'InternetIsBeautiful', 
                  'Jokes', 'history', 'gadgets', 'food', 'photoshopbattles', 'Futurology', 
                  'Documentaries', 'dataisbeautiful', 'GetMotivated', 'UpliftingNews', 
                  'listentothis', 'philosophy', 'OldSchoolCool', 'Art', 'creepy', 'nosleep',
                  'WritingPrompts', 'TwoXChromosomes', 'Fitness']
    
    # test function
    for i in input_list:
        sub_start = time.time()
        import_data(input_address + i, output_address, subreddit = subreddits, print_flag = False)
        print('\n' + i + ' completed!!!')
        print('month run time: ' + str( round( time.time() - sub_start, 1) ) )
        
    print('run_time: ' + str( round( time.time()-main_start ,1) ) + 's')