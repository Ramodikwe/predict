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
def dictionary_of_metrics(arr):

    #defination of variables
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    n = len(arr)
    arr=sorted(arr)
    maximum=0
    avg=0
    std=0
    var=0
    #calculation of the mean

    while a < n:
        avg += arr[a]
        a += 1
    avg = avg / n
    avg = round(avg,2)

    #calculation of standard deviation

    while b < n:
        std += (arr[b] - avg )** 2
        b += 1
    std = (std / (n-1)) ** 0.5
    std=round(std,2)

    #calculation of varience

    while e < n:
        var += (arr[e] - avg )** 2
        e += 1
    var = (var / (n-1))
    var=round(var,2)


    #calculation of median

    if ((n/2)-0.5)== (n//2):
        med=arr[int((n/2)+0.5)]
    else:
        med=arr[int((n/2)-0.5)] + arr[int((n/2)+0.5)]
        med=med/2


    #calculation of  max

    while c<n:
        if maximum<=arr[c]:
            maximum=arr[d]
        c+=1

    #calculation of minimum

    minimum =maximum
    while d < n:
        if minimum >= arr[d]:
            minimum = arr[d]
            minimum=round(minimum)
        d += 1

    met_dict={'mean':avg,'median':med,'varience':var,'standard deviation':std,'min':minimum,'max':maximum}
    return met_dict



### END FUNCTION


# In[6]:


dictionary_of_metrics(gauteng)


# _**Expected Output**_:
#
# ```python
# dictionary_of_metrics(gauteng) == {'mean': 26244.42,
#                                    'median': 24403.5,
#                                    'var': 108160153.17,
#                                    'std': 10400.01,
#                                    'min': 8842.0,
#                                    'max': 39660.0}
#  ```

# ## Function 2: Five Number Summary
#
# **Function Specifications:**
# - The function takes a list as input.
# - The function  returns a `dict` with keys `'max'`, `'median'`, `'min'`, `'q1'`, and `'q3'` corresponding to the maximum, median, minimum, first quartile and third quartile, respectively. You may use numpy functions to aid in your calculations.
# - All numerical values are rounded to two decimal places.

# In[64]:


### START FUNCTION
def five_num_summary(arr):
    #defination of variables
    c=0
    d=0
    maximum=0
    n=len(arr)
    #first quartile
    arr=sorted(arr)
    quartile_1=np.quantile(arr,0.25)
    quartile_1=round(quartile_1,2)

    #third quartile
    quartile_3=np.quantile(arr,0.75)
    quartile_3=round(quartile_3,2)

    #calculation of  max

    while c<n:
        if maximum<=arr[c]:
            maximum=arr[c]
        c+=1

    #calculation of minimum

    minimum =maximum
    while d < n:
        if minimum >= arr[d]:
            minimum = arr[d]
            minimum=round(minimum,2)
        d += 1

    #calculation of median

    if ((n/2)-0.5)== (n//2):
        med=arr[int((n/2)+0.5)]
    else:
        med=arr[int((n/2)-0.5)] + arr[int((n/2)+0.5)]
        med=med/2

    dict_five_num={'max':maximum,'median':med,'min':minimum,'q1':quartile_1,'q3':quartile_3}
    return dict_five_num



### END FUNCTION


# In[65]:


five_num_summary(gauteng)


# _**Expected Output:**_
#
# ```python
# five_num_summary(gauteng) == {
#     'max': 39660.0,
#     'median': 24403.5,
#     'min': 8842.0,
#     'q1': 18653.0,
#     'q3': 36372.0
# }
#
# ```

# ## Function 3: Date Parser
#
# The `dates` variable (created at the top of this notebook) is a list of dates represented as strings. The string contains the date in `'yyyy-mm-dd'` format, as well as the time in `hh:mm:ss` formamt. The first three entries in this variable are:
# ```python
# dates[:3] == [
#     '2019-11-29 12:50:54',
#     '2019-11-29 12:46:53',
#     '2019-11-29 12:46:10'
# ]
# ```
#
# **Function Specifications:**
# - The function takes a list of strings as input.
# - Each string in the input list is formatted as `'yyyy-mm-dd hh:mm:ss'`.
# - The function returns a list of strings where each element in the returned list contains only the date in the `'yyyy-mm-dd'` format.

# In[9]:


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
    twitter_list=list(twitter_df['Tweets']) #creating a list from the tweets column in the twitter df
    n=0
    split_tweets=list(range(len(twitter_df)))
    while n<len(twitter_df):
        split_tweets[n]=(twitter_list[n].split())#split every element in the twitter list
        split_tweets[n]=[x.lower() for x in split_tweets[n]] #convert the tweets to lower case
        words=split_tweets[n]
        similar=list(set(stop_words_dict['stopwords']) & set(words)) # create a list from a set of similar words from the twitter split list and the stowpword dictionary
        m=0
        while m<len(similar): #loop to remove every word in the similar list from the split tweets list
            words.remove(similar[m])
            m+=1
        n+=1
    twitter_df['Without Stop Words']=split_tweets #return the twitter list without the stopwords
    return twitter_df

### END FUNCTION


# In[62]:


stop_words_remover(twitter_df.copy())


# _**Expected Output**_:
#
# Specific rows:
#
# ```python
# stop_words_remover(twitter_df.copy()).loc[0, "Without Stop Words"] == ['@bongadlulane', 'send', 'email', 'mediadesk@eskom.co.za']
# stop_words_remover(twitter_df.copy()).loc[100, "Without Stop Words"] == ['#eskomnorthwest', '#mediastatement', ':', 'notice', 'supply', 'interruption', 'lichtenburg', 'area', 'https://t.co/7hfwvxllit']
# ```
#
# Whole table:
# ```python
# stop_words_remover(twitter_df.copy())
# ```
#
# > <table class="dataframe" border="1">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>Tweets</th>
#       <th>Date</th>
#       <th>Without Stop Words</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>@BongaDlulane Please send an email to mediades...</td>
#       <td>2019-11-29 12:50:54</td>
#       <td>[@bongadlulane, send, email, mediadesk@eskom.c...</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>@saucy_mamiie Pls log a call on 0860037566</td>
#       <td>2019-11-29 12:46:53</td>
#       <td>[@saucy_mamiie, pls, log, 0860037566]</td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td>@BongaDlulane Query escalated to media desk.</td>
#       <td>2019-11-29 12:46:10</td>
#       <td>[@bongadlulane, query, escalated, media, desk.]</td>
#     </tr>
#     <tr>
#       <th>3</th>
#       <td>Before leaving the office this afternoon, head...</td>
#       <td>2019-11-29 12:33:36</td>
#       <td>[leaving, office, afternoon,, heading, weekend...</td>
#     </tr>
#     <tr>
#       <th>4</th>
#       <td>#ESKOMFREESTATE #MEDIASTATEMENT : ESKOM SUSPEN...</td>
#       <td>2019-11-29 12:17:43</td>
#       <td>[#eskomfreestate, #mediastatement, :, eskom, s...</td>
#     </tr>
#     <tr>
#       <th>...</th>
#       <td>...</td>
#       <td>...</td>
#       <td>...</td>
#     </tr>
#     <tr>
#       <th>195</th>
#       <td>Eskom's Visitors Centresâ€™ facilities include i...</td>
#       <td>2019-11-20 10:29:07</td>
#       <td>[eskom's, visitors, centresâ€™, facilities, incl...</td>
#     </tr>
#     <tr>
#       <th>196</th>
#       <td>#Eskom connected 400 houses and in the process...</td>
#       <td>2019-11-20 10:25:20</td>
#       <td>[#eskom, connected, 400, houses, process, conn...</td>
#     </tr>
#     <tr>
#       <th>197</th>
#       <td>@ArthurGodbeer Is the power restored as yet?</td>
#       <td>2019-11-20 10:07:59</td>
#       <td>[@arthurgodbeer, power, restored, yet?]</td>
#     </tr>
#     <tr>
#       <th>198</th>
#       <td>@MuthambiPaulina @SABCNewsOnline @IOL @eNCA @e...</td>
#       <td>2019-11-20 10:07:41</td>
#       <td>[@muthambipaulina, @sabcnewsonline, @iol, @enc...</td>
#     </tr>
#     <tr>
#       <th>199</th>
#       <td>RT @GP_DHS: The @GautengProvince made a commit...</td>
#       <td>2019-11-20 10:00:09</td>
#       <td>[rt, @gp_dhs:, @gautengprovince, commitment, e...</td>
#     </tr>
#   </tbody>
# </table>

# In[ ]:
