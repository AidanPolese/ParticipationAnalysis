import copy
import csv
import json
import operator
import re
import random
from pprint import pprint as pp
import pandas as pd
from sklearn.decomposition import PCA
from adjustText import adjust_text

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import STOPWORDS, WordCloud

# list of topics decided as being important
# extracted using code but then further hand cleaned
IMPORTANT_TOPICS = ['institutions', 'services', 'platforms', 'video', 'support', 'medical', 'bias', 'world', 'courses', 'processing', 'rdf', 'industry', 'cloud', 'gapminder', 'networks', 'network', 'dataset', 'datasets', 'application', 'environment', 'healthcare', 'unstructured', 'structured', 'analytical', 'statistical', 'predictive', 'personalization', 'personal', 'dashboards', 'company', 'visualization', 'management', 'applications', 'quantitative', 'statistics', 'security', 'qualitative',
                    'online', 'companies', 'users', 'models', 'software', 'technology', 'learners', 'systems', 'access', 'system', 'techniques', 'research', 'privacy', 'la', 'student', 'user', 'edm', 'business', 'methods', 'internet', 'devices', 'knowledge', 'students', 'education', 'tool', 'web', 'iot', 'educational', 'analysis', 'tools', 'mining', 'analytics', 'learning', 'data', 'type functionality', 'functionality analytics', 'information retrieval', 'visualizations dashboards',
                    'sematic web', 'data structure', 'graph databases', 'insider intelligence', 'data users', 'operating costs', 'iot sensors', 'iot applications', 'education organizations', 'internet connection', 'remote learning', 'medical field', 'medical professionals', 'data sharing', 'iot education', 'wearable iot', 'self learning', 'future iot', 'security data', 'analyzing data', 'types data', 'human judgement', 'optimizing learning', 'collection data', 'analytics process',
                    'online courses', 'user interface', 'learning materials', 'data collection', 'analytics measurement', 'insights data', 'predictive models', 'mining tools', 'privacy concerns', 'learning algorithms', 'analyze data', 'supply chain', 'vector', 'volume', 'velocity', 'data scientist', 'life insurance', 'quantitative analytics', 'analytic tools', 'google data', 'data visualizations', 'lesson plans', 'internet education', 'digital campus', 'iot data', 'role learning',
                    'neural networks', 'educational institutions', 'education system', 'student behavior', 'society learning', 'data generated', 'business decisions', 'learning platforms', 'statistical methods', 'education sector', 'students learning', 'student engagement', 'academic analytics', 'patient care', 'pre processing', 'qualitative analysis', 'data lake', 'access data', 'insurance companies', 'diagnostic analytics', 'statistical analysis', 'open source', 'data management',
                    'connected devices', 'knowledge discovery', 'education data', 'social network', 'reporting data', 'learners contexts', 'critical thinking', 'improving learning', 'improve learning', 'processing data', 'improve student', 'data education', 'learning outcomes', 'type data', 'semi structured', '', 'social media', 'structured data', 'analytical tools', 'science data', 'learning data', 'google trends', 'data analysts', 'data set', 'prescriptive analytics', 'ibm cognos',
                    'network analysis', 'artificial intelligence', 'teaching', 'educational settings', 'edm la', 'analytics research', 'personal data', 'google trend', 'e learning', 'analysis tools', 'collection analysis', 'analytics data', 'data source', 'tools systems', 'analysis data', 'learning experience', 'data scientists', 'unstructured data', 'user friendly', 'analytics tools', 'web data', 'data sets', 'mining techniques', 'learning environments', 'learning styles', 'data points',
                    'mining edm', 'business intelligence', 'sentiment analysis', 'predictive analytics', 'data visualization', 'data driven', 'health care', 'learning environment', 'decision making', 'data sources', 'higher education', 'data collected', 'quantitative data', 'data science', 'qualitative data', 'iot devices', 'data analytics', 'data analysis', 'internet things', 'learning knowledge', 'knowledge analytics', 'machine learning', 'open data', 'linked data', 'semantic web',
                    'educational data', 'learning analytics', 'big data', 'data mining', 'education data mining', 'society learning analytics', 'processing of data', 'collection analysis reporting', 'reporting of data', 'web of data', 'linked open data', 'web linked data', 'learning environments occurs', 'learning analytics research', 'big data analytics', 'data mining techniques', 'mining educational data', 'internet of things', 'educational data mining']

# lists of words that are excluded from the important words and phrases lists
RESTRICTED_LIST = ['of', 'the', 'at', 'there', 'some', 'my', 'be', 'use', 'her', 'than', 'and', 'this', 'an', 'would', 'first', 'a', 'have', 'each', 'make', 'water', 'to', 'from', 'which', 'like', 'been', 'in', 'or', 'she', 'him', 'call', 'is', 'one', 'do', 'into', 'who', 'you', 'had', 'how', 'time', 'oil', 'that', 'by', 'their', 'has', 'its', 'it', 'word', 'if', 'look', 'now',
                   'he', 'but', 'will', 'two', 'find', 'was', 'not', 'up', 'more', 'long', 'for', 'what', 'other', 'write', 'down', 'on', 'all', 'about', 'go', 'day', 'are', 'were', 'out', 'see', 'did', 'as', 'we', 'many', 'number', 'get', 'with', 'when', 'then', 'no', 'come', 'his', 'your', 'them', 'way', 'made', 'they', 'can', 'these', 'could', 'may', 'i', 'said', 'so', 'people', 'part', 'am', 'very', 'really', 't', 's']
RESTRICTED_LIST_NO_OF = ['the', 'at', 'there', 'some', 'my', 'be', 'use', 'her', 'than', 'and', 'this', 'an', 'would', 'first', 'a', 'have', 'each', 'make', 'water', 'to', 'from', 'which', 'like', 'been', 'in', 'or', 'she', 'him', 'call', 'is', 'one', 'do', 'into', 'who', 'you', 'had', 'how', 'time', 'oil', 'that', 'by', 'their', 'has', 'its', 'it', 'word', 'if', 'look', 'now',
                         'he', 'but', 'will', 'two', 'find', 'was', 'not', 'up', 'more', 'long', 'for', 'what', 'other', 'write', 'down', 'on', 'all', 'about', 'go', 'day', 'are', 'were', 'out', 'see', 'did', 'as', 'we', 'many', 'number', 'get', 'with', 'when', 'then', 'no', 'come', 'his', 'your', 'them', 'way', 'made', 'they', 'can', 'these', 'could', 'may', 'i', 'said', 'so', 'people', 'part', 'am', 'very', 'really', 't', 's']


def checkReplies(replySection, week, wordCountsPerWeek):
    # recursively check all reply sections to extract their content
    for reply in replySection:
        person = reply[6:]
        wordsInReply = len(replySection[reply]['content'].split())
        wordCountsPerWeek[week][person]['replies'] += wordsInReply
        if replySection[reply]['replies']:
            subReplies = replySection[reply]['replies']
            checkReplies(subReplies, week, wordCountsPerWeek)


def averageWordCountsPerWeek(wordCountsTotal):
    # get the average word counts per week per person
    outDict = {}
    for person in wordCountsTotal:
        outDict[person] = round(wordCountsTotal[person] / 6)
    return outDict


def getTotalWordCounts(wordCountsPerWeek, wordCountsTotal):
    # get the total word counts per week per person
    for week in wordCountsPerWeek:
        for person in wordCountsPerWeek[week]:
            totalWordsAtWeek = wordCountsPerWeek[week][person]['posts'] + wordCountsPerWeek[week][person]['replies']
            wordCountsTotal[person] += totalWordsAtWeek


def getAverageWordCountsPostsPerWeek(wordCountsPerWeek, averageWordCountsPostPerWeek):
    # get the total word counts per post per person per week (lots of pers)
    for week in wordCountsPerWeek:
        for person in wordCountsPerWeek[week]:
            averageWordCountsPostPerWeek[person] += wordCountsPerWeek[week][person]['posts']
    for person in averageWordCountsPostPerWeek:
        averageWordCountsPostPerWeek[person] = round(averageWordCountsPostPerWeek[person] / 6)


def createBarGraphWordCounts(wordCountsTotal, averageWordCountsPerWeekTotal, averageWordCountsPostPerWeek):
    # covers xi, xii
    # creates bar graphs of total and average word counts
    # lists to be graphed being extracted from dictionary containers
    people = []
    totalWordsList = []
    avgWordsPerWeekList = []
    avgWordsPerPostList = []
    for person in wordCountsTotal:
        people.append(person)
        totalWordsList.append(wordCountsTotal[person])
        avgWordsPerWeekList.append(averageWordCountsPerWeekTotal[person])
        avgWordsPerPostList.append(averageWordCountsPostPerWeek[person])
    # create graph for average words
    x = np.arange(len(people))
    width = 0.3
    fig, ax = plt.subplots()
    rects2 = ax.bar(x + width/2, avgWordsPerWeekList, width, label='Average Words Per Week')
    rects3 = ax.bar(x - width/2, avgWordsPerPostList, width, label='Average Words Per Post')
    ax.set_title("Average Word Contributions")
    ax.set_xlabel("People")
    ax.set_ylabel("Number of Words")
    ax.set_xticks(x, people)
    ax.legend()
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    fig.tight_layout()
    # create graph for total word counts
    fig2, ax2 = plt.subplots()
    rects1 = ax2.bar(x, totalWordsList, width, label='Total Words')
    ax2.set_title("Total Word Contributions")
    ax2.set_xlabel("People")
    ax2.set_ylabel("Number of Words")
    ax2.set_xticks(x, people)
    ax2.bar_label(rects1, padding=3)
    fig2.tight_layout()


def numberOfTopicsCovered(topicCounts):
    # count the number of topics being covered
    totalTopicsCovered = {}
    totalImportantTopicsCovered = {}
    # extract the topic information from dictionary containers into lists to be graphed
    for person in topicCounts:
        topicsCovered = 0
        importantTopicsCovered = 0
        for topic in topicCounts[person]:
            if topicCounts[person][topic]:
                topicsCovered += 1
                if topic in IMPORTANT_TOPICS:
                    importantTopicsCovered += 1
        totalTopicsCovered[person] = topicsCovered
        totalImportantTopicsCovered[person] = importantTopicsCovered
    people = []
    # assemble lists of information
    topicsCoveredList = []
    importantTopicsCoveredList = []
    for person in totalImportantTopicsCovered:
        people.append(person)
        topicsCoveredList.append(totalTopicsCovered[person])
        importantTopicsCoveredList.append(totalImportantTopicsCovered[person])
    # graph information
    x = np.arange(len(people))
    width = 0.3
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, topicsCoveredList, width, label='Total Topics Covered')
    rects2 = ax.bar(x + width/2, importantTopicsCoveredList, width, label='Total Important Topics Covered')
    ax.set_ylabel('Topic Number')
    ax.set_title("Topics Covered")
    ax.set_xticks(x, people)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig.tight_layout()


def createBarGraphReplyCount(repliesPerPerson):
    # covers xiv
    # graph the number of replies per person from all weeks
    people = []
    numberOfReplies = []
    # extract from dictionary
    for person in repliesPerPerson:
        people.append(person)
        numberOfReplies.append(repliesPerPerson[person])
    fig, ax = plt.subplots()
    ax.set_title('Replies Per Person')
    ax.set_ylabel("Number of Replies")
    ax.set_xlabel("People")
    rects = ax.bar(people, numberOfReplies)
    ax.set_xticks(people)
    ax.bar_label(rects, padding=3)
    fig.tight_layout()
    # plt.show()


def clusterPeopleWhoMentionTopic(topicCounts):
    # extract people, the topics they mention and their mention counts from dictionary container
    clusterList = []
    people = []
    for person in topicCounts:
        people.append(person)
        row = []
        for topic in topicCounts[person]:
            row.append(topicCounts[person][topic])
        clusterList.append(row)
    # feed information into matrix
    clusterDF = pd.DataFrame(clusterList)
    # initialize sctructure to reduce the dimensional features from a few hundered to 2 for graphing and clustering
    pca = PCA()
    pcax = pca.fit_transform(clusterDF)
    # create scatter plot graph
    fig, ax = plt.subplots()
    ax.set_title('Total Topics Mentioned')
    fig.tight_layout()
    ax.scatter(pcax[:, 0], pcax[:, 5])
    # adjust texts to not overlap
    texts = [plt.text(pcax[:, 0][i], pcax[:, 5][i], people[i]) for i in range(len(pcax[:, 0]))]
    adjust_text(texts, only_move={'points': 'y', 'texts': 'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    # repeat but only using the 245 important topics and phrases
    # cluster using important topics
    clusterList = []
    people = []
    for person in topicCounts:
        people.append(person)
        row = []
        for topic in topicCounts[person]:
            if topic in IMPORTANT_TOPICS:
                row.append(topicCounts[person][topic])
        clusterList.append(row)
    clusterDF = pd.DataFrame(clusterList)
    pca = PCA()
    pcax = pca.fit_transform(clusterDF)
    fig2, ax2 = plt.subplots()
    fig2.tight_layout()
    ax2.scatter(pcax[:, 0], pcax[:, 5])
    ax2.set_title('Total Important Topics Mentioned')
    texts = [plt.text(pcax[:, 0][i], pcax[:, 5][i], people[i]) for i in range(len(pcax[:, 0]))]
    adjust_text(texts, only_move={'points': 'y', 'texts': 'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))


def getWordPairs(content):
    # go through all content ever posted by a students and extract all two word pair couples
    wordPairs = {}
    for wordIndex in range(0, len(content)-1):
        currentPair = content[wordIndex] + ' ' + content[wordIndex+1]
        if currentPair not in wordPairs:
            wordPairs[currentPair] = 1
        else:
            wordPairs[currentPair] += 1
    sortedPairs = sorted(wordPairs.items(), key=operator.itemgetter(1))
    return sortedPairs


def getWordTriplets(content):
    # go through all content ever posted by a students and extract all three word phrases
    wordTriples = {}
    for wordIndex in range(0, len(content)-2):
        currentTriple = content[wordIndex] + ' ' + content[wordIndex+1] + ' ' + content[wordIndex+2]
        if currentTriple not in wordTriples:
            wordTriples[currentTriple] = 1
        else:
            wordTriples[currentTriple] += 1
    sortedTriples = sorted(wordTriples.items(), key=operator.itemgetter(1))
    return sortedTriples


def getReplyContent(replySection, week, content):
    # recursively go through replies to get their content
    for reply in replySection:
        content.extend(replySection[reply]['content'].split())
        if replySection[reply]['replies']:
            subReplies = replySection[reply]['replies']
            getReplyContent(subReplies, week, content)


def getAllContent(discussion):
    # go through all posts and replies to retrieve their content
    content = []
    for week in discussion:
        for postPerson in discussion[week]:
            if discussion[week][postPerson]['content'] != "No submission":
                content.extend(discussion[week][postPerson]['content'].split())
                if discussion[week][postPerson]['replies']:
                    replySection = discussion[week][postPerson]['replies']
                    getReplyContent(replySection, week, content)
    return content


def mostPopularTopics(postsJson):
    # initialize all information structures
    allWords = {}
    content = ''
    # get all content ever posted from students
    content = getAllContent(postsJson)
    content = ' '.join(content)
    # remove all non-alphabetical symbols
    content = re.sub(r'[^\w]|[0-9]|\_', ' ', content)
    content = content.lower()
    splitContent = content.split()
    cleanedSplitContentPairs = []
    # sort through all content ever posted into lists that contain only important keywords and phrases
    for word in splitContent:
        if word not in RESTRICTED_LIST:
            cleanedSplitContentPairs.append(word)
    cleanedSplitContentTriples = []
    for word in splitContent:
        if word not in RESTRICTED_LIST_NO_OF:
            cleanedSplitContentTriples.append(word)
    for word in splitContent:
        if word not in allWords and word not in RESTRICTED_LIST:
            allWords[word] = 1
        elif word in allWords and word not in RESTRICTED_LIST:
            allWords[word] += 1
    # extract popular clean words
    popularWords = sorted(allWords.items(), key=operator.itemgetter(1))
    # extract popular clean words, 2 word and 3 word phrases
    wordPairs = getWordPairs(cleanedSplitContentPairs)
    wordTriples = getWordTriplets(cleanedSplitContentTriples)
    return popularWords, wordPairs, wordTriples


def changeToStringFrequency(popularWordsTupleList):
    # create string containing repitions of words and phrases to be turned into word clouds
    outString = ''
    listOfStrings = []
    for i in range(0, len(popularWordsTupleList)):
        word = popularWordsTupleList[i][0]
        appearances = popularWordsTupleList[i][1]
        tempString = ''
        for j in range(0, appearances):
            tempString += ' '
            tempString += word
            listOfStrings.append(word)
        outString += tempString
    return outString, listOfStrings


def generateWordClouds(popularWords, popularPairs, popularTriples):
    # generate word clouds
    # popular single words
    popularWordsString, listOfStrings = changeToStringFrequency(popularWords)
    random.shuffle(listOfStrings)
    popularWordsString = " ".join(listOfStrings)
    stopwords = set(STOPWORDS)
    popularWordCloud = WordCloud(width=800, height=800, background_color='white', stopwords=stopwords, min_font_size=10).generate(popularWordsString)
    fig, ax = plt.subplots()
    ax.imshow(popularWordCloud)
    ax.axis('off')
    fig.tight_layout(pad=0)
    # popular pairs
    # The one I used in the report
    popularWordsString, listOfStrings = changeToStringFrequency(popularPairs)
    random.shuffle(listOfStrings)
    popularWordsString = " ".join(listOfStrings)
    stopwords = set(STOPWORDS)
    popularWordCloud = WordCloud(width=800, height=800, background_color='white', stopwords=stopwords, min_font_size=10).generate(popularWordsString)
    fig, ax = plt.subplots()
    ax.imshow(popularWordCloud)
    ax.axis('off')
    fig.tight_layout(pad=0)
    # popular triples
    popularWordsString, listOfStrings = changeToStringFrequency(popularTriples)
    random.shuffle(listOfStrings)
    popularWordsString = " ".join(listOfStrings)
    stopwords = set(STOPWORDS)
    popularWordCloud = WordCloud(width=800, height=800, background_color='white', stopwords=stopwords, min_font_size=10).generate(popularWordsString)
    fig, ax = plt.subplots()
    ax.imshow(popularWordCloud)
    ax.axis('off')
    fig.tight_layout(pad=0)


def getTopicMentions(topics, topicCounts, perPersonContent):
    # get the amount of times a person mentions a key topic / word / phrase
    for person in perPersonContent:
        for topic in topics:
            topicMentions = perPersonContent[person].count(topic)
            topicCounts[person][topic] = topicMentions


def createReplyNetworkCSVDictBase(replySection, week, repliesToWho, currentPerson):
    # get who is involved in the same post
    for reply in replySection:
        replyPerson = reply[6:]
        if replyPerson != currentPerson:
            repliesToWho[currentPerson][replyPerson] += 1
            # maybe next line, causes a bit of noise
            # repliesToWho[replyPerson][currentPerson] += 1
        if replySection[reply]['replies']:
            subReplies = replySection[reply]['replies']
            createReplyNetworkCSVDictBase(subReplies, week, repliesToWho, currentPerson)


def createReplyNetworkCSVStructure(repliesToWho):
    # create structure to hold who made a post and who replied to it
    # to be fed later into Netlytic
    outList = [['date', 'from', 'to']]
    for postPerson in repliesToWho:
        for replyPerson in repliesToWho[postPerson]:
            if repliesToWho[postPerson][replyPerson]:
                # maybe reverse for different looking structure
                # outList.append([postPerson, replyPerson])
                outList.append(['2/1/2011', replyPerson, postPerson])
    return outList


def countReplies(replySection, week, repliesPerPerson):
    # goes through all replies ever to count how many were made per person
    for reply in replySection:
        person = reply[6:]
        repliesPerPerson[person] += 1
        if replySection[reply]['replies']:
            subReplies = replySection[reply]['replies']
            countReplies(subReplies, week, repliesPerPerson)


def getPersonContentReplies(replySection, week, perPersonContent):
    # get all content contained in a reply
    for reply in replySection:
        person = reply[6:]
        perPersonContent[person] += ' '
        perPersonContent[person] += replySection[reply]['content']
        if replySection[reply]['replies']:
            subReplies = replySection[reply]['replies']
            getPersonContentReplies(subReplies, week, perPersonContent)


def combinePostsWeek(wordCounts):
    outDict = copy.deepcopy(wordCounts)
    for week in wordCounts:
        for postPerson in wordCounts[week]:
            if postPerson.endswith(('-2', '-3')):
                searchPerson = postPerson[:-2]
                for matchPerson in wordCounts[week]:
                    if matchPerson == searchPerson:
                        outDict[week][searchPerson]['posts'] += wordCounts[week][postPerson]['posts']
                        del(outDict[week][postPerson])
    return outDict


def buildReplyNetwork(discussions):
    # build dictionary to hold information to be fed into Netlytic
    outDict = {}
    for week in discussions:
        for postPerson in discussions[week]:
            person = postPerson[5:]
            if person.endswith(('-2', '-3')):
                person = person[:-2]
            if person not in outDict:
                outDict[person] = {}
    allPeople = outDict.keys()
    for person in outDict:
        for personKey in allPeople:
            if person != personKey:
                outDict[person][personKey] = 0
    return outDict


def buildTotalWordCounts(discussions):
    # build dictionary to hold total word counts per person
    outDict = {}
    for week in discussions:
        for postPerson in discussions[week]:
            person = postPerson[5:]
            if person.endswith(('-2', '-3')):
                person = person[:-2]
            if person not in outDict:
                outDict[person] = 0
    return outDict


def buildPerPersonContent(discussions):
    # build dictionary to hold all content from posts and replies made by a person
    outDict = {}
    for week in discussions:
        for postPerson in discussions[week]:
            person = postPerson[5:]
            if person.endswith(('-2', '-3')):
                person = person[:-2]
            if person not in outDict:
                outDict[person] = ''
    return outDict


def buildWordCounts(discussions):
    # build dictionary to hold word counts of all students
    outDict = {}
    for week in discussions:
        outDict[week] = {}
        for postPerson in discussions[week]:
            person = postPerson[5:]
            outDict[week][person] = {'posts': 0, 'replies': 0}
    return outDict


def buildTopicCounts(discussions, wordPairs, wordTriples):
    # build dictionary to hold all keywords, and key phrases made by students
    outDict = {}
    for week in discussions:
        for postPerson in discussions[week]:
            person = postPerson[5:]
            if person.endswith(('-2', '-3')):
                person = person[:-2]
            if person not in outDict:
                outDict[person] = {}
    for person in outDict:
        for wordPairTuple in wordPairs:
            wordPair = wordPairTuple[0]
            outDict[person][wordPair] = 0
        for wordTripleTuple in wordTriples:
            wordTriple = wordTripleTuple[0]
            outDict[person][wordTriple] = 0
    return outDict


def dataAnalysis():
    with open("Discussion.json", 'r', encoding='utf-8') as inFile:
        # load in the discussion json information
        postsJson = json.load(inFile)
        # covers xi
        # covers xii
        # create the dictionary to hold each students total word contributions to the course
        wordCountsPerWeek = buildWordCounts(postsJson)
        # get word counts each week for both posts and replies
        for week in postsJson:
            for postPerson in postsJson[week]:
                if postsJson[week][postPerson]['content'] != "No submission":
                    person = postPerson[5:]
                    wordsAtWeekPosts = len(postsJson[week][postPerson]['content'].split())
                    wordCountsPerWeek[week][person]['posts'] += wordsAtWeekPosts
                    if postsJson[week][postPerson]['replies']:
                        replySection = postsJson[week][postPerson]['replies']
                        # recursively check all reply sections to extract their content
                        checkReplies(replySection, week, wordCountsPerWeek)
        wordCountsPerWeek = combinePostsWeek(wordCountsPerWeek)
        wordCountsTotal = buildTotalWordCounts(postsJson)
        getTotalWordCounts(wordCountsPerWeek, wordCountsTotal)
        # get the average word counts per week of each student
        averageWordCountsPerWeekTotal = averageWordCountsPerWeek(wordCountsTotal)
        averageWordCountsPostPerWeek = buildTotalWordCounts(postsJson)
        getAverageWordCountsPostsPerWeek(wordCountsPerWeek, averageWordCountsPostPerWeek)
        # graph word counts each week
        createBarGraphWordCounts(wordCountsTotal, averageWordCountsPerWeekTotal, averageWordCountsPostPerWeek)
        # covers xiii
        # build dictionary containing who replies to a person's post on the forums
        repliesToWho = buildReplyNetwork(postsJson)
        for week in postsJson:
            for postPerson in postsJson[week]:
                if postsJson[week][postPerson]['content'] != "No submission":
                    person = postPerson[5:]
                    if postsJson[week][postPerson]['replies']:
                        replySection = postsJson[week][postPerson]['replies']
                        if person.endswith(('-2', '-3')):
                            person = person[:-2]
                        createReplyNetworkCSVDictBase(replySection, week, repliesToWho, person)
        # write out the matrix as a csv file to be fed into Netlytic
        repliesToWhoCSV = createReplyNetworkCSVStructure(repliesToWho)
        with open("PostReplyNetworks.csv", 'w', encoding='utf-8', newline='') as outFile:
            writer = csv.writer(outFile)
            writer.writerows(repliesToWhoCSV)
        # covers xiv
        # count the amount of replies a person has made in total
        repliesPerPerson = buildTotalWordCounts(postsJson)
        for week in postsJson:
            for postPerson in postsJson[week]:
                if postsJson[week][postPerson]['content'] != "No submission":
                    person = postPerson[5:]
                    if postsJson[week][postPerson]['replies']:
                        replySection = postsJson[week][postPerson]['replies']
                        countReplies(replySection, week, repliesPerPerson)
        # graph the number of replies per person
        createBarGraphReplyCount(repliesPerPerson)
        # covers xvii
        # create lists of the most popular words that are used by people in discussions to be used to create word clouds
        popularWords, wordPairs, wordTriples = mostPopularTopics(postsJson)
        generateWordClouds(popularWords, wordPairs, wordTriples)
        # covers xv
        # create the structure to hold how many times a person mentions a specific topic
        topicCounts = buildTopicCounts(postsJson, wordPairs, wordTriples)
        perPersonContent = buildPerPersonContent(postsJson)
        # calculate how many times a person mentions a topic
        for week in postsJson:
            for postPerson in postsJson[week]:
                if postsJson[week][postPerson]['content'] != "No submission":
                    person = postPerson[5:]
                    if person.endswith(('-2', '-3')):
                        person = person[:-2]
                    perPersonContent[person] += ' ' + postsJson[week][postPerson]['content']
                    if postsJson[week][postPerson]['replies']:
                        replySection = postsJson[week][postPerson]['replies']
                        getPersonContentReplies(replySection, week, perPersonContent)
        # get the total number of topics that are used
        topics = topicCounts['aidanpo1'].keys()
        # total the numbers of topics mention by each person
        getTopicMentions(topics, topicCounts, perPersonContent)
        # code used to check my own personal topic mentions
        # for topic in topicCounts['aidanpo1']:
        #     if topicCounts['aidanpo1']:
        #         if topic in IMPORTANT_TOPICS and topicCounts['aidanpo1'][topic] > 1:
        #             print(topic, ':', topicCounts['aidanpo1'][topic])
        # create scatter plots showing clusters of people who mention the same topics
        clusterPeopleWhoMentionTopic(topicCounts)
        # covers xvi
        numberOfTopicsCovered(topicCounts)


def main():
    # update matplotlib parameters
    params = {'legend.fontsize': 20, 'axes.titlesize': 25, 'axes.labelsize': 25, 'xtick.labelsize': 20}
    params = {'legend.fontsize': 20, 'axes.titlesize': 25, 'axes.labelsize': 25}
    plt.rcParams.update(params)
    dataAnalysis()
    # display all plots
    plt.show()


main()
