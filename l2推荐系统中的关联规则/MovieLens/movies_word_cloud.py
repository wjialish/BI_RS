#词云展示
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
#去掉停用词
def remove_stop_words(f):
    stop_words=['Movie','aaa']
    for stop_word in stop_words:
        f=f.replace(stop_word,'')
    return f


#生成词云
def create_word_cloud(f):
    print('根据词频，生成词云！')
    f = remove_stop_words(f)
    cut_text=word_tokenize(f)
    print("cut_text1----")
    print(cut_text)
    '''
    cut_text1----
['Toy', 'Story', '(', '1995', ')', 'Jumanji', '(', '1995', ')', 'Grumpier', 'Old', 'Men', '(', '1995', ')', 'Waiting', 'to', 'Exhale', '(', '1995', ')', 'Father', 'of', 'the', 'Bride', 'Part', 'II', '(', '1995', ')', 'Heat', '(', '1995', ')', 'Sabrina', '(', '1995', ')', 'Tom', 'and', 'Huck', '(', '1995', ')', 'Sudden', 'Death', '(', '1995', ')', 'GoldenEye', '(', '1995', ')', 'Adventure|Animation|Children|Comedy|Fantasy', 'Adventure|Children|Fantasy', 'Comedy|Romance', 'Comedy|Drama|Romance', 'Comedy', 'Action|Crime|Thriller', 'Comedy|Romance', 'Adventure|Children', 'Action', 'Action|Adventure|Thriller']
    '''
    cut_text=" ".join(cut_text)
    print("cut_text2-----")
    print(cut_text)
    '''
    cut_text2-----
Toy Story ( 1995 ) Jumanji ( 1995 ) Grumpier Old Men ( 1995 ) Waiting to Exhale ( 1995 ) Father of the Bride Part II ( 1995 ) Heat ( 1995 ) Sabrina ( 1995 ) Tom and Huck ( 1995 ) Sudden Death ( 1995 ) GoldenEye ( 1995 ) Adventure|Animation|Children|Comedy|Fantasy Adventure|Children|Fantasy Comedy|Romance Comedy|Drama|Romance Comedy Action|Crime|Thriller Comedy|Romance Adventure|Children Action Action|Adventure|Thriller

    '''
    wc=WordCloud(
        max_words=100,
        width=2000,
        height=1200,
    )
    wordcloud=wc.generate(cut_text)
    print("wordcloud-----")
    print(wordcloud)
    '''
    wordcloud-----
<wordcloud.wordcloud.WordCloud object at 0x1a17d74f60>
    '''
    #写词云图片
    wordcloud.to_file("worldcloud.jpg")
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()



#数据加载
data=pd.read_csv('./movies2.csv')
#读取title和genres
print(data['title'])
'''
0                      Toy Story (1995)
1                        Jumanji (1995)
2               Grumpier Old Men (1995)
3              Waiting to Exhale (1995)
4    Father of the Bride Part II (1995)
5                           Heat (1995)
6                        Sabrina (1995)
7                   Tom and Huck (1995)
8                   Sudden Death (1995)
9                      GoldenEye (1995)
'''
print(data['genres'])
'''
0    Adventure|Animation|Children|Comedy|Fantasy
1                     Adventure|Children|Fantasy
2                                 Comedy|Romance
3                           Comedy|Drama|Romance
4                                         Comedy
5                          Action|Crime|Thriller
6                                 Comedy|Romance
7                             Adventure|Children
8                                         Action
9                      Action|Adventure|Thriller
'''
title=" ".join((data['title']))
genres=" ".join((data['genres']))
print("title-----")
print(title)
'''
title-----
Toy Story (1995) Jumanji (1995) Grumpier Old Men (1995) Waiting to Exhale (1995) Father of the Bride Part II (1995) Heat (1995) Sabrina (1995) Tom and Huck (1995) Sudden Death (1995) GoldenEye (1995)
'''
print("genres-----")
print(genres)
'''
genres-----
Adventure|Animation|Children|Comedy|Fantasy Adventure|Children|Fantasy Comedy|Romance Comedy|Drama|Romance Comedy Action|Crime|Thriller Comedy|Romance Adventure|Children Action Action|Adventure|Thriller
'''
# title=data['title']
# genres=" ".join(data['genres'])
all_word=title+genres
print("all_word----")
print(all_word)
'''
genres-----
Adventure|Animation|Children|Comedy|Fantasy Adventure|Children|Fantasy Comedy|Romance Comedy|Drama|Romance Comedy Action|Crime|Thriller Comedy|Romance Adventure|Children Action Action|Adventure|Thriller
all_word----
Toy Story (1995) Jumanji (1995) Grumpier Old Men (1995) Waiting to Exhale (1995) Father of the Bride Part II (1995) Heat (1995) Sabrina (1995) Tom and Huck (1995) Sudden Death (1995) GoldenEye (1995)Adventure|Animation|Children|Comedy|Fantasy Adventure|Children|Fantasy Comedy|Romance Comedy|Drama|Romance Comedy Action|Crime|Thriller Comedy|Romance Adventure|Children Action Action|Adventure|Thriller

'''
#生成词云
create_word_cloud(all_word)