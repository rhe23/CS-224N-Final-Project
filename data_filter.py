'''
Author: Tyler Chase
Date: 2017/02/28

This function takes in the address of a input json file, the address of an output json file, the subreddit you
want filtered, the upvotes that you want the posts to exceed, and a boolean
value as to whether you want the outputs printed to the console. It then 
returns a json file with the titles of the subreddits, it's respective sub, 
and the total number of upvotes (one post per line)
''' 


import json 
import time
import numpy as np

def import_data(input_address, output_address, outputs = np.inf, subreddit = None, upvotes = 0, print_flag = False):
    # Open the file to be filtered
    f = open(input_address, "r")
    # Open the output file to be written to
    f_o = open(output_address, "a")
    num_outputs = 0 
    read_errors = 0
    # Import JSON lines one by one
    for line in f:
        # Do while less than max desired outputs
        output = {}  # output line
        if num_outputs < outputs:
            # try to convert from json to python dictionary
            try:
                temp = json.loads(line)
                # If subreddit input is None don't filter by subreddit
                if subreddit == None:
                    if temp['ups'] >= upvotes:
                        # Form output, convert to json and write to output
                        output['subreddit'] = temp['subreddit']
                        output['ups'] = temp['ups']
                        output['title'] = temp['title']
                        output = json.dumps(output)
                        # If print_flag is true print to console
                        if print_flag:
                            print('subreddit:')
                            print(temp['subreddit'])
                            print('upvotes: ')
                            print(temp['ups'])
                            print('title')
                            print(temp['title:'])
                            print('\n')
                        f_o.write(output)
                        f_o.write('\n')
                        num_outputs+=1
                else:        
                    if temp['ups'] >= upvotes and subreddit == temp['subreddit']:
                        # Form output, convert to json and write to output
                        output['subreddit'] = temp['subreddit']
                        output['ups'] = temp['ups']
                        output['title'] = temp['title']
                        output = json.dumps(output)
                        # If print_flag is true print to console
                        if print_flag:
                            print('subreddit:')
                            print(temp['subreddit'])
                            print('upvotes: ')
                            print(temp['ups'])
                            print('title:')
                            print(temp['title'])
                            print('\n')
                        f_o.write(output)
                        f_o.write('\n')
                        num_outputs+=1
            except:
                read_errors+=1
    f.close()
    f_o.close()
    print('decoding errors: ' + str(read_errors))
    print('posts created:' + str(num_outputs))
    
# Test filter function
if __name__ == "__main__":
    
    start_time = time.time()

    input_address = "/Users/tylerchase/Documents/Stanford_Classes/CS224n_Natural_Language_Processing_with_Deep_Learning/final project/data/RS_2015-09"
    output_address = "/Users/tylerchase/Documents/Stanford_Classes/CS224n_Natural_Language_Processing_with_Deep_Learning/final project/data/test_output"

    import_data(input_address, output_address, outputs = 1000, upvotes = 100, subreddit = "tifu", print_flag = False)
    
    print('run_time: ' + str( round(time.time()-start_time,1) ) + 's')
