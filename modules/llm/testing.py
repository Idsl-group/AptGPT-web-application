from skbio import DNA
import pandas as pd
from skbio.alignment import local_pairwise_align_ssw


def check_alignment(seq1, seq2):
    _, score, _ = local_pairwise_align_ssw(DNA(seq1), DNA(seq2))
    return score / (len(seq2) + len(seq1))


def test_sequences(data):
    top_20_sequences = pd.read_csv("/home/rtulluri/DAPTEV_Model/data/start/guan11_top_100_sequences.csv").head(20)
    data['key'] = 0
    top_20_sequences['key'] = 0
    final_data = pd.merge(top_20_sequences, data, how='outer', on='key', suffixes=('_org', '_gen')).drop(columns='key')
    final_data['Alignment Scores'] = final_data[['Sequences', 'Aptamers']].apply(
        lambda x: check_alignment(x['Sequences'], x['Aptamers']), axis=1)
    final_data.sort_values(by=['Alignment Scores'], ascending=False, inplace=True)
    final_data.drop_duplicates(subset=['Aptamers'], inplace=True)
    final_data['Alignment Scores'] = final_data['Alignment Scores'].apply(lambda x: str(round(x * 100, 2)) + "%")
    
    return final_data[['Aptamers', 'Alignment Scores']].reset_index(drop=True)

def compare_sequences(data, target):
    data['key'] = 0
    target['key'] = 0
    final_data = pd.merge(target, data, how='outer', on='key', suffixes=('_org', '_gen')).drop(columns='key')
    print(final_data.columns)
    final_data['Alignment Scores'] = final_data[['Sequences_org', 'Sequences_gen']].apply(
        lambda x: check_alignment(x['Sequences_org'], x['Sequences_gen']), axis=1)
    final_data.sort_values(by=['Alignment Scores'], ascending=False, inplace=True)
    final_data.drop_duplicates(subset=['Sequences_gen'], inplace=True)
    final_data['Alignment Scores'] = final_data['Alignment Scores'].apply(lambda x: str(round(x * 100, 2)) + "%")
    final_data.rename(columns={'Sequences_gen': 'Aptamers'}, inplace=True)
    
    return final_data[['Aptamers', 'Alignment Scores']].reset_index(drop=True).head(20)