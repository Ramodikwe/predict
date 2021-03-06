#Functions to anlyse skom data given
# ## Imports

# In[1]:


import pandas as pd
import numpy as np


# ## Data Loading and Preprocessing

# ### Electricification by province (EBP) data

# In[2]:


ebp_url = 'https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/electrification_by_province.csv'
ebp_df = pd.read_csv(ebp_url)

for col, row in ebp_df.iloc[:,1:].iteritems():
    ebp_df[col] = ebp_df[col].str.replace(',','').astype(int)

ebp_df.head()


# ### Twitter data

# In[3]:


twitter_url = 'https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/twitter_nov_2019.csv'
twitter_df = pd.read_csv(twitter_url)
twitter_df.head()


# ## Important Variables (Do not edit these!)

# In[4]:


# gauteng ebp data as a list
gauteng = ebp_df['Gauteng'].astype(float).to_list()

# dates for twitter tweets
dates = twitter_df['Date'].to_list()

# dictionary mapping official municipality twitter handles to the municipality name
mun_dict = {
    '@CityofCTAlerts' : 'Cape Town',
    '@CityPowerJhb' : 'Johannesburg',
    '@eThekwiniM' : 'eThekwini' ,
    '@EMMInfo' : 'Ekurhuleni',
    '@centlecutility' : 'Mangaung',
    '@NMBmunicipality' : 'Nelson Mandela Bay',
    '@CityTshwane' : 'Tshwane'
}

# dictionary of english stopwords
stop_words_dict = {
    'stopwords':[
        'where', 'done', 'if', 'before', 'll', 'very', 'keep', 'something', 'nothing', 'thereupon',
        'may', 'why', 'â€™s', 'therefore', 'you', 'with', 'towards', 'make', 'really', 'few', 'former',
        'during', 'mine', 'do', 'would', 'of', 'off', 'six', 'yourself', 'becoming', 'through',
        'seeming', 'hence', 'us', 'anywhere', 'regarding', 'whole', 'down', 'seem', 'whereas', 'to',
        'their', 'various', 'thereafter', 'â€˜d', 'above', 'put', 'sometime', 'moreover', 'whoever', 'although',
        'at', 'four', 'each', 'among', 'whatever', 'any', 'anyhow', 'herein', 'become', 'last', 'between', 'still',
        'was', 'almost', 'twelve', 'used', 'who', 'go', 'not', 'enough', 'well', 'â€™ve', 'might', 'see', 'whose',
        'everywhere', 'yourselves', 'across', 'myself', 'further', 'did', 'then', 'is', 'except', 'up', 'take',
        'became', 'however', 'many', 'thence', 'onto', 'â€˜m', 'my', 'own', 'must', 'wherein', 'elsewhere', 'behind',
        'becomes', 'alone', 'due', 'being', 'neither', 'a', 'over', 'beside', 'fifteen', 'meanwhile', 'upon', 'next',
        'forty', 'what', 'less', 'and', 'please', 'toward', 'about', 'below', 'hereafter', 'whether', 'yet', 'nor',
        'against', 'whereupon', 'top', 'first', 'three', 'show', 'per', 'five', 'two', 'ourselves', 'whenever',
        'get', 'thereby', 'noone', 'had', 'now', 'everyone', 'everything', 'nowhere', 'ca', 'though', 'least',
        'so', 'both', 'otherwise', 'whereby', 'unless', 'somewhere', 'give', 'formerly', 'â€™d', 'under',
        'while', 'empty', 'doing', 'besides', 'thus', 'this', 'anyone', 'its', 'after', 'bottom', 'call',
        'nâ€™t', 'name', 'even', 'eleven', 'by', 'from', 'when', 'or', 'anyway', 'how', 'the', 'all',
        'much', 'another', 'since', 'hundred', 'serious', 'â€˜ve', 'ever', 'out', 'full', 'themselves',
        'been', 'in', "'d", 'wherever', 'part', 'someone', 'therein', 'can', 'seemed', 'hereby', 'others',
        "'s", "'re", 'most', 'one', "n't", 'into', 'some', 'will', 'these', 'twenty', 'here', 'as', 'nobody',
        'also', 'along', 'than', 'anything', 'he', 'there', 'does', 'we', 'â€™ll', 'latterly', 'are', 'ten',
        'hers', 'should', 'they', 'â€˜s', 'either', 'am', 'be', 'perhaps', 'â€™re', 'only', 'namely', 'sixty',
        'made', "'m", 'always', 'those', 'have', 'again', 'her', 'once', 'ours', 'herself', 'else', 'has', 'nine',
        'more', 'sometimes', 'your', 'yours', 'that', 'around', 'his', 'indeed', 'mostly', 'cannot', 'â€˜ll', 'too',
        'seems', 'â€™m', 'himself', 'latter', 'whither', 'amount', 'other', 'nevertheless', 'whom', 'for', 'somehow',
        'beforehand', 'just', 'an', 'beyond', 'amongst', 'none', "'ve", 'say', 'via', 'but', 'often', 're', 'our',
        'because', 'rather', 'using', 'without', 'throughout', 'on', 'she', 'never', 'eight', 'no', 'hereupon',
        'them', 'whereafter', 'quite', 'which', 'move', 'thru', 'until', 'afterwards', 'fifty', 'i', 'itself', 'nâ€˜t',
        'him', 'could', 'front', 'within', 'â€˜re', 'back', 'such', 'already', 'several', 'side', 'whence', 'me',
        'same', 'were', 'it', 'every', 'third', 'together'
    ]
}


# ## Function 1: Metric Dictionary
#
# **Function Specifications:**
# - Function allows a list as input.
# - It  returns a `dict` with keys `'mean'`, `'median'`, `'std'`, `'var'`, `'min'`, and `'max'`, corresponding to the mean, median, standard deviation, variance, minimum and maximum of the input list, respectively.
# - The standard deviation and variance values must be unbiased. **Hint:** use the `ddof` parameter in the corresponding numpy functions!
# - All values in the returned `dict` are rounded to 2 decimal places.

# In[5]:


### START FUNCTION
def dictionary_of_metrics(items):
    """ Calculates mean, median, variance, standard deviation, min and max
    from a given list

        Parameters
        ----------
        items: list
                The values on which to perfom calculations.

        Returns
        -------
        metrics: dictionary
                    Dictionary of metrics calculated
    """

    # Initializing dictionary
    metrics = {'mean':0,
               'median':0,
               'var':0,
               'std':0,
               'min':0,
               'max':0}

    #Calculating metrics and assigning values
    metrics['mean'] = round(sum(items)/len(items),2)
    metrics['median'] = round(np.median(items),2)
    metrics['var'] = round(np.var(items, ddof=1),2)
    metrics['std'] = round(np.std(items, ddof=1),2)
    metrics['min'] = round(min(items),2)
    metrics['max'] = round(max(items),2)
    return metrics

### END FUNCTION


### START FUNCTION
def five_num_summary(items):
    """Calculates max, median, min, quartile 1 and quartile 3
    from a list of numbers

    Parameters
    ----------
    items: list
            List of numbers o which to perform operations

    Returns
    -------
    fives: dictionary
            The resulting summary after operations have been performed
    """

    # Initializing dictionary
    fives = {'max':0,
             'median':0,
             'min':0,
             'q1':0,
             'q3':0}

    # Performing operations and assigning to dictionary
    fives['max'] = round(max(items),2)
    fives['median'] = round(np.median(items),2)
    fives['min'] = round(min(items),2)
    fives['q1'] = np.quantile(items, 0.25)
    fives['q3'] = np.quantile(items, 0.75)

    return fives

### END FUNCTION


### START FUNCTION
def date_parser(dates):
    only_date=list(range(len(dates)))       #creating a list of numbers of asize equivalent to the lenghth of the twitter dataframe
    n=0
    while n <len(dates):
        date_split=dates[n].split()         #spliting the dates column so that the times and dates could be separate
        only_date[n]=date_split[0]          #taking the first element of the split dates column which will be the dates only and leaving the times
        n+=1
    return only_date                        #returning only the dates
### END FUNCTION


# In[10]:


date_parser(dates[:3])


# _**Expected Output:**_
#
# ```python
# date_parser(dates[:3]) == ['2019-11-29', '2019-11-29', '2019-11-29']
# date_parser(dates[-3:]) == ['2019-11-20', '2019-11-20', '2019-11-20']
# ```

# ## Function 4: Municipality & Hashtag Detector
#

# **Function Specifications:**
# * Function  takes a pandas `dataframe` as input.
# * Extract the municipality from a tweet using the `mun_dict` dictonary given below, and insert the result into a new column named `'municipality'` in the same dataframe.
# * Uses the entry `np.nan` when a municipality is not found.
# * Extracts a list of hashtags from a tweet into a new column named `'hashtags'` in the same dataframe.
# * Uses the entry `np.nan` when no hashtags are found.
#

# ```

# In[11]:


### START FUNCTION
def extract_municipality_hashtags(twitter_df):
    twitter_list=list(twitter_df['Tweets'])
    #The etraction of manucipality
    Municipality=list(range(len(twitter_df)))       #creating a lits of numbers equivalent to the lenghth of the twitter df
    for n in range(len(twitter_df)):                #initializing the minucipality list with np.nan
        Municipality[n]=np.nan
    n=1
    while n<=len(twitter_df):                       #while loop to replace the np.nan in the list for tweets that have municipality tags on them
        to_compare=twitter_list[n-1].split()
        i=1
        while i<=len(to_compare):
            for (key, value) in mun_dict.items():
                if key == to_compare[i-1]:          #if the municipality tag is found , the np.nan is replace with a value from the mun_dict dictonary
                    Municipality[n-1]=value
                    i=len(to_compare)+1
                    break
            i+=1
        n+=1
    twitter_df['municipality']=Municipality         #adding the municipality column to the twitter DataFrame
    #the extraction of Hashtag
    Hashtag=list(range(len(twitter_df)))
    for n in range(0,len(twitter_df)):              #initializing the hashtags list with np.NaN
        Hashtag[n]=np.nan
    n=1
    check='#'                                       #hashtag variable to compare with expression as target:
        pass
    while n<=len(twitter_df):                       #while loop to replace the np.nan in the list for tweets that have  hashtags on them
        i=1
        to_compare=twitter_list[n-1].split()
        temp = [position for position in to_compare if position[0] == check] #if a hasthag is found the strijg of that hashtag is put in a list , so if one or more hasthags are on a single they'll all be stored and put in the hastag lists
        Hashtag[n-1]=temp
        Hashtag[n-1]=[x.lower() for x in Hashtag[n-1]]
        if len(Hashtag[n-1])==0:
            Hashtag[n-1]=np.nan
        n+=1
    twitter_df['hashtags']=Hashtag                 #adding the hashtag list of lists to the twitterr DataFrame as a column
    return twitter_df

### END FUNCTION



# ## Function 5: Number of Tweets per Day
#
# **Function Specifications:**
# - It should take a pandas dataframe as input.
# - It should return a new dataframe, grouped by day, with the number of tweets for that day.
# - The index of the new dataframe should be named `Date`, and the column of the new dataframe should be `'Tweets'`, corresponding to the date and number of tweets, respectively.
# - The date should be formated as `yyyy-mm-dd`, and should be a datetime object. **Hint:** look up `pd.to_datetime` to see how to do this.

# In[13]:


### START FUNCTION
def number_of_tweets_per_day(df):
    n=0
    while n<len(twitter_df):
        dates = twitter_df['Date'].to_list()        #creating a list of dates from the twitter DataFrame
        only_date=list(range(len(dates)))           #creating a list of numbers with a lenghth corresponding to the twitter df lenghth
        n=0
        while n <len(dates):                        #a loop that puts dates at which the tweets were tweeted
            date_split=dates[n].split()
            only_date[n]=date_split[0]
            n+=1
        n+=1
    set_date=sorted(list(set(only_date)))           #sorting and also removing duplicates from the dates list created above and that will be used as a counter
    count=[only_date.count(m) for m in set_date ]   #counting the number of times a date appear in the date list
    df=pd.DataFrame()                               #creating a DataFrame to hold the values [dates and number of number_of_tweets_per_day]
    set_date=pd.to_datetime(set_date)               #converting the dates to to_date data type
    df['Date']=set_date                             #adding the dates  column to the created dataframe
    df['Tweets']=count                              #adding the number_of_tweets_per_day column to the dataframe
    df=df.set_index('Date')
    return df

### END FUNCTION


# # Function 6: Word Splitter
#
# **Function Specifications:**
# - It should take a pandas dataframe as an input.
# - The dataframe should contain a column, named `'Tweets'`.
# - The function should split the sentences in the `'Tweets'` into a list of seperate words, and place the result into a new column named `'Split Tweets'`. The resulting words must all be lowercase!
# - The function should modify the input dataframe directly.
# - The function should return the modified dataframe.

# In[15]:


### START FUNCTION
def word_splitter(twitter_df):
#decleration of twitter df and other attributes used in the function
    twitter_list=list(twitter_df['Tweets'])
    n=0
    split_tweets=list(range(len(twitter_df)))
    while n<len(twitter_df):                                            #loop to split all the elements in the tweets column in the twitter dataframe
        split_tweets[n]=(twitter_list[n].split())
        split_tweets[n]=[x.lower() for x in split_tweets[n]]            #convert the tweets to lower case
        n+=1
        twitter_df['Split Tweets']=split_tweets
    return twitter_df                                                   #return the split tweets
### END FUNCTION


# # Function 7: Stop Words
#
# **Function Specifications:**
# - It  takes a pandas dataframe as input.
# - It tokenise the sentences according to the definition in function 6. Note that function 6 **cannot be called within this function**.
# - It remove all stop words in the tokenised list. The stopwords are defined in the `stop_words_dict` variable defined at the top of this notebook.
# - The resulting tokenised list is placed in a column named `"Without Stop Words"`.
# - The function  modifies the input dataframe.
# - The function returns the modified dataframe.
#

# In[61]:


### START FUNCTION
def stop_words_remover(df):
    """Removes stop words from tweet and appends new column to DataFrame with operation results.

    Parameters
    ----------
    df: DataFrame
        DataFrame on which to perform operation

    Returns
    -------
    df: DataFrame
        Modified DataFrame with 'Split Tweets' column added
    """

    # Stop word remover function
    def remover(tweet):
        """Removes stop words from a string.

        Parameters
        ----------
        tweet: string
            Tweet on which operaton is performed

        Returns
        -------
        without_stops: list
            List of words not found in stop_words_dict
        """

        # Lower and split tweet and initialize empty list
        split_tweet = tweet.lower().split()
        without_stops = []

        # For each word in the split tweet
        for word in split_tweet:

            # If the word is not in the dictionary...
            if word not in stop_words_dict['stopwords']:
                without_stops.append(word) # append to list
        return without_stops

    # Add new column to DataFrame
    df['Without Stop Words'] = df['Tweets'].apply(remover)
    return df
### END FUNCTION
