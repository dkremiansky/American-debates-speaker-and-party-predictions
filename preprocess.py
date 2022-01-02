import pandas as pd
import numpy as np
import os
import re
import nltk.data
import nltk
import string
from nltk.tokenize import word_tokenize
from pandas import Series
import pickle


def isEnglish(s):  # checks if a character is an english character
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def find_speaker(row):
    text = row['tokenized_data']
    for word in text:
        if word == '.':
            return ' '.join(text[:text.index('.')]), text[text.index('.') + 1:]
        if not word.isupper():
            return '', text
    return '', text


democratic_candidates = ['Kennedy', 'Carter', 'Mondale', 'Ferraro', 'Dukakis', 'Bentsen', 'Clinton', 'Gore',
                             'Lieberman', 'Kerry', 'Edwards', 'Obama', 'Biden', 'Kaine', 'Harris']
republican_candidates = ['Nixon', 'Ford', 'Dole', 'Reagan', 'Bush', 'Quayle', 'Kemp', 'Cheney', 'McCain',
                         'Palin', 'Romney', 'Ryan', 'Trump', 'Pence']


def party_association(row):
    spaeker_last_name = row['speaker'].split()[-1].strip('.')
    if spaeker_last_name.lower() in [name.lower() for name in democratic_candidates]:
        return 'democrat'
    elif spaeker_last_name.lower() in [name.lower() for name in republican_candidates]:
        return 'republican'
    else :
        return 'other'


def text_line_files_to_text_lines_dfs(files_data_list):
    text_lines_dfs = []
    for debate in files_data_list:
        for i in range(len(debate) - 1, -1, -1):
            # split text rows by ':' or ';' chars and create dfs
            # first part is considered as speaker and the rest is content
            debate[i] = re.split(':|;', debate[i], maxsplit=1)
            # delete empty rows
            if i % 2:
                debate.pop(i)
        debate_df = pd.DataFrame.from_records(debate, columns=['speaker', 'transcript'])
        text_lines_dfs.append(debate_df)
    return text_lines_dfs


def split_paragraph_to_sentences(df):
    df_split = pd.DataFrame({col: np.repeat(df[col].values, df['sentence'].str.len())
                        for col in df.columns.difference(['sentence'])}).assign(
        **{'sentence': np.concatenate(df['sentence'].values)})[df.columns.tolist()]
    return df_split


president_by_year = {1976: 'Ford',1980:'Carter',1984:'Reagan',1988:'Reagan',1992:'Bush',1996:'Clinton',
                     2000:'Clinton',2004:'Bush',2008:'Bush',2012:'Obama',2016:'Obama',2020:'Trump'}


def give_THE_PRESIDENT_his_name(speaker, debate_index, years_of_that_debate_type):
    if 'the president' in speaker.lower():
        speaker = president_by_year[years_of_that_debate_type[debate_index]]  # replace 'THE PRESIDENT' with his name
    return speaker


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')  # must run before: nltk.download('punkt')


def read_data(path): # todo!!!!!
    """
    creates a list of dataframes of the debates
    :param path: path to a directory which contains text transcripts of all debates
        Return:
          - debates_dfs: list of dataframes (columns=['speaker','sentence','tokenized','speaker_changed'])
    The list may be pickled
    """
    file_formats = {1:[0,1,2,3,6,8,11,14,15,21,23,24,25,26,44,45], 2:[13,17,28,31,32,33,34,36,39,40,41,42,43],
                    3:[4, 5, 7, 9], 4:[10,12,16,18,19,20,22,27,29,30,35,37,38],
                    'with not relevant introduction': [-2,-1], 'type_2_special': 41}
    texts = []
    years = []
    # read data and debate year of each file
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    text = f.read()
                    year = int(text.split('\n')[0].split()[-1])
                    texts.append(text)
                    years.append(year)
    # split debates texts to paragraphs (lines of text are usually paragraphs)
    files_data = [line.split('\n')[1:] for line in texts]
    # remove not relevant introduction
    for i in file_formats['with not relevant introduction']:
        files_data[i] = files_data[i][7:]
    # split files and years by debate type
    type_1_dfs_years = [years[i] for i in file_formats[1]]
    files_data_type_1 = [files_data[i] for i in file_formats[1]]
    type_2_dfs_years = [years[i] for i in file_formats[2]]
    files_data_type_2 = [files_data[i] for i in file_formats[2]]
    type_3_dfs_years = [years[i] for i in file_formats[3]]
    files_data_type_3 = [files_data[i] for i in file_formats[3]]
    type_4_dfs_years = [years[i] for i in file_formats[4]]
    files_data_type_4 = [files_data[i] for i in file_formats[4]]

    '''type 1'''
    dfs_type_1 = []
    text_lines_dfs_type_1 = text_line_files_to_text_lines_dfs(files_data_type_1)
    for j, df in enumerate(text_lines_dfs_type_1):
        df['sentence'] = [[] for _ in range(len(df))]
        # find speaker
        previous_speaker = ''
        for i, row in df.iterrows():
            # convert row to a list of sentences
            row['sentence'] = tokenizer.tokenize(row['transcript']) if isinstance(row['transcript'],str) else []
            # check if the speaker attribute isn't really the speaker
            if isinstance(row['transcript'],str) and\
                    ((not row['speaker'].isupper() and 'Mc' not in row['speaker']) or len(tokenizer.tokenize(row['speaker'])) > 1):
                row['sentence'] = [row['speaker']] + row['sentence']
                row['speaker'] = previous_speaker
            previous_speaker = row['speaker']
            row['speaker'] = give_THE_PRESIDENT_his_name(row['speaker'], j, type_1_dfs_years)

        df_type_1 = split_paragraph_to_sentences(df)
        df_type_1['tokenized'] = df_type_1.apply(lambda row: word_tokenize(row['sentence'].lower()), axis=1)
        # find if speaker changed
        df_type_1['speaker_changed'] = [[] for _ in range(len(df_type_1))]
        previous_speaker = ''
        for i, row in df_type_1.iterrows():
            if i > 0:
                row['speaker_changed'] = True if row['speaker'] != previous_speaker and not row['speaker'].split()[-1] in previous_speaker else False
            if i == 0:
                row['speaker_changed'] = True
            previous_speaker = row['speaker']
        df_type_1.drop('transcript', axis=1, inplace=True)
        dfs_type_1.append(df_type_1)

    '''type 2'''
    text_lines_dfs_type_2 = []
    dfs_type_2 = []
    # remove introductions
    for debate in files_data_type_2[2:7]:  # todo variable names instead of numbers
        index = files_data_type_2.index(debate)
        debate = debate[2:]
        files_data_type_2[index] = debate
    for debate in files_data_type_2[8:]:  # todo
        index = files_data_type_2.index(debate)
        debate = debate[7:]
        files_data_type_2[index] = debate

    for debate in files_data_type_2:
        for i in range(len(debate) - 1, -1, -1):
            # split text rows by ':' or ';' chars and create dfs
            # first part is considered as speaker and the rest is content
            debate[i] = re.split(':|;', debate[i], maxsplit=1)
            # remove empty lines
            if i % 2 and files_data_type_2.index(debate) != 10:  # todo
                debate.pop(i)
            if i % 2 == 0 and files_data_type_2.index(debate) == 10:
                debate.pop(i)
        for i in range(len(debate) - 1, -1, -1):
            if len(debate[i]) != 2:
                debate[i-1][len(debate[i-1])-1] += debate[i][0]
                debate.pop(i)
        debate_df = pd.DataFrame.from_records(debate, columns=['speaker', 'transcript'])
        text_lines_dfs_type_2.append(debate_df)
    for j, df2 in enumerate(text_lines_dfs_type_2):
        df2['sentence'] = [[] for _ in range(len(df2))]
        previous_speaker = ''
        for i, row in df2.iterrows():
            row['sentence'] = tokenizer.tokenize(row['transcript']) if isinstance(row['transcript'],str) else []
            if isinstance(row['transcript'],str) and\
                    ((not row['speaker'].isupper() and 'Mc' not in row['speaker'])\
                     or len(tokenizer.tokenize(row['speaker'])) > 1):
                row['sentence'] = [row['speaker']] + row['sentence']
                row['speaker'] = previous_speaker
            previous_speaker = row['speaker']
            row['speaker'] = give_THE_PRESIDENT_his_name(row['speaker'], j, type_2_dfs_years)
        df_type_2 = split_paragraph_to_sentences(df2)

        df_type_2['tokenized'] = df_type_2.apply(lambda row: word_tokenize(row['sentence'].lower()), axis=1)
        df_type_2['speaker_changed'] = [[] for _ in range(len(df_type_2))]
        previous_speaker = ''
        for i, row in df_type_2.iterrows():
            if i > 0:
                row['speaker_changed'] = True if row['speaker'] != previous_speaker\
                and not row['speaker'].split()[-1] in previous_speaker else False
            if i == 0:
                row['speaker_changed'] = True
            previous_speaker = row['speaker']
        df_type_2.drop('transcript', axis=1, inplace=True)

        df_type_2 = df_type_2[df_type_2.sentence != '.']
        df_type_2 = df_type_2[df_type_2.sentence != ' .']
        dfs_type_2.append(df_type_2)

    '''type 3'''
    text_lines_dfs_type_3 = text_line_files_to_text_lines_dfs(files_data_type_3)
    dfs_type3 = []
    for j, df_debate in enumerate(text_lines_dfs_type_3[0:]):
        df_debate.rename(columns={'speaker': 'data'}, inplace=True)
        df_debate['tokenized_data'] = df_debate.apply(lambda row: word_tokenize(row['data']), axis=1)
        df_debate['speaker_text'] = [tuple()] * df_debate.shape[0]
        df_debate['speaker'] = ''
        df_debate['tokenized'] = [[]] * df_debate.shape[0]
        df_debate['speaker_changed'] = 1
        for i, row in df_debate.iterrows():
            row['speaker_text'] = find_speaker(row)
            row['speaker'], row['tokenized'] = row['speaker_text']
            row['speaker_changed'] = True if row['speaker'] else False
            df_debate.iloc[i, :] = row
        dfs_type3.append(df_debate)

    dfs_type_3 = []
    for j, df_type3 in enumerate(dfs_type3):
        df_type_3 = df_type3.filter(['speaker','data','tokenized','speaker_changed'], axis=1)
        previous_speaker = ''
        for i, row in df_type_3.iterrows():
            if re.search('[a-zA-Z]', row['speaker']) == None:
                row['speaker'] = previous_speaker
            else:
                previous_speaker = row['speaker']
                row['data'] = ' '.join(row['data'].split()[2:])
            row['speaker'] = give_THE_PRESIDENT_his_name(row['speaker'], j, type_3_dfs_years)
        df_type_3.columns = ['speaker', 'sentence', 'tokenized', 'speaker_changed']
        dfs_type_3.append(df_type_3)

    '''type 4'''
    dfs_type_4 = []
    for j, debate in enumerate(files_data_type_4):
        for i in range(len(debate)-1,-1,-1):
            tokenized = tokenizer.tokenize(debate[i])
            if i%2 or not tokenized:
                debate.pop(i)
        debate_to_df = []
        for i in range(len(debate)):
            try:
                tokenized = tokenizer.tokenize(debate[i])
            except:
                continue
            speaker = tokenized[0]
            speaker_for_df = give_THE_PRESIDENT_his_name(speaker,j, type_4_dfs_years)
            speaker_tokenized = word_tokenize(speaker)
            if (len(tokenized)==1 and speaker_tokenized[-1] not in string.punctuation and isEnglish(speaker_tokenized[-1])) \
                    or (len(tokenized)>1 and debate[i][-1] not in string.punctuation and isEnglish(debate[i][-1])):  # considered as title
                print(f'considered as title: {debate[i]}')
            elif len(tokenized)>2 and (speaker_tokenized)[0].lower() == 'gov'\
                    and len(word_tokenize(tokenized[1])) <= 3:
                # first 2 sentences are considered as speaker: speaker+' '+tokenized[1]
                debate_to_df.append([speaker+' '+tokenized[1], tokenized[2], word_tokenize(tokenized[2]), True])
                for s in tokenized[3:]:
                    debate_to_df.append([speaker+' '+tokenized[1], s, word_tokenize(s), False])
            elif speaker_tokenized[0] == 'Q.':
                debate_to_df.append(['Q.', speaker[2:], speaker_tokenized[2:], True])  # 'Q.: question's content...'
                for s in tokenized[1:]:
                    debate_to_df.append(['Q.', s, word_tokenize(s), False])
            elif len(tokenized)>1 and len(speaker_tokenized) <= 4 and speaker_tokenized[-1] == '.' and\
                    not any(w[0].islower() for w in speaker_tokenized):
                # first sentence is the speakers name, the rest is the speech'
                debate_to_df.append([speaker_for_df, tokenized[1], word_tokenize(tokenized[1]), True])
                for s in tokenized[2:]:
                    debate_to_df.append([speaker_for_df, s, word_tokenize(s), False])
            else:
                # considered as text
                for s in tokenized:
                    debate_to_df.append([debate_to_df[-1][0], s, word_tokenize(s), False])

        debate_df = pd.DataFrame.from_records(debate_to_df,columns=['speaker','sentence','tokenized','speaker_changed'])
        dfs_type_4.append(debate_df)

    debates_dfs = dfs_type_1 + dfs_type_2 + dfs_type_3 + dfs_type_4

    for i,df in enumerate(debates_dfs):
        # replace non-english characters
        df['sentence'] = df.apply(lambda row: ''.join([c * isEnglish(c) + ' ' * (1 - isEnglish(c)) for c in row['sentence']]).strip('- '),
                 axis=1)
        df.drop(columns=['tokenized'], inplace=True)

    # add democrat or republican label
    for debate in debates_dfs:
        debate['party'] = debate.apply(party_association,axis = 1)

    return debates_dfs

if __name__ == '__main__':
    path = 'debates'
    debates_dfs = read_data(path)
    with open('debates_dfs_fixed.pkl', 'wb') as f:
        pickle.dump(debates_dfs,f)

