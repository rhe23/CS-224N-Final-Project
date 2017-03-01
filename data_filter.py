'''
Author: Tyler Chase
Date: 2017/02/28

This function takes in the address of a input json file, the address of an output 
json files, the date of the input file, a list of subreddits you want filtered, 
and a boolean value as to whether you want the outputs printed to the console. 
It then returns a json file with the titles of the subreddits, it's respective sub, 
and the total number of upvotes (one post per line)
''' 

# import libraries
import json 
import time

# this function sorts a list of post dictionaries by the number of upvotes
# then it returns the highest "num_posts" posts
def sort(dict_list, num_posts):
    key = lambda x: x['ups']
    temp = sorted(dict_list, key = key, reverse = True)
    return(temp[:num_posts])

def import_data(input_address, output_address, date, outputs = None, subreddit = None, print_flag = False):
    if subreddit == None:
        raise NameError('Please enter a list of subreddits')
    if outputs == None:
        raise NameError('Please enter desired number of top outputs')
    # Open the file to be filtered
    f = open(input_address, "r")
    # Open the output file to be written to
    output_files = []
    num_outputs_list = []
    output_listOfLists = []
    for i in subreddit:
        output_files.append( open(output_address + i + '_' + date + '_raw', "a") )
        num_outputs_list.append(0)
        output_listOfLists.append([])
        
       
    #f_o = open(output_address, "a")
    read_errors = 0
    ascci_errors = 0
    # Import JSON lines one by one
    posts_seen = 0
    for line in f:
        posts_seen+=1
        # Do while less than max desired outputs
        output = {}  # output line
        try:
            temp = json.loads(line)
            sub_iter = 0
            for sub in subreddit:
                if sub == temp['subreddit']:
                    # try to see if the title can be converted into ascci characters
                    try:
                        str(temp['title'])
                        # Form output, convert to json and write to output
                        output['subreddit'] = temp['subreddit']
                        output['ups'] = temp['ups']
                        output['title'] = temp['title']
                        #output = json.dumps(output)
                        # If print_flag is true print to console
                        if print_flag:
                            print('subreddit:')
                            print(temp['subreddit'])
                            print('upvotes: ')
                            print(temp['ups'])
                            print('title:')
                            print(temp['title'])
                            print('\n')
                        output_listOfLists[sub_iter].append(output)
                        num_outputs_list[sub_iter]+=1
                    except:
                        ascci_errors+=1
                sub_iter+=1
        except:
            read_errors+=1
    f.close()

    # Sort files and then output to output json files    
    for i in range( len(subreddit) ):
        temp = sort(output_listOfLists[i], outputs)
        for j in temp:
            j = json.dumps(j)
            output_files[i].write(j)
            output_files[i].write('\n')
            
    # close all output files
    for i in range( len(subreddit) ):
        output_files[i].close()
        
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
    
    start_time = time.time()
    
    # input and output addresses
    input_address = "/Users/tylerchase/Documents/Stanford_Classes/CS224n_Natural_Language_Processing_with_Deep_Learning/final project/data/RS_2015-09"
    output_address = "/Users/tylerchase/Documents/Stanford_Classes/CS224n_Natural_Language_Processing_with_Deep_Learning/final project/data/"
    
    # subreddit list
    # listed with similar subs on each line
    subreddits = ['AskReddit', 'LifeProTips',\
                  'nottheonion', 'news', 'science',\
                  'trees', \
                  'tifu', \
                  'personalfinance', \
                  'mildlyinteresting', 'interestingasfuck']
    
    # test function
    import_data(input_address, output_address, '201509', outputs = 1000, subreddit = subreddits, print_flag = False)
        
    print('run_time: ' + str( round(time.time()-start_time,1) ) + 's')
