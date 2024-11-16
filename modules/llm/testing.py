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
    print(final_data)
    final_data['alignment_score'] = final_data[['Sequences', 'Aptamers']].apply(
        lambda x: check_alignment(x['Sequences'], x['Aptamers']), axis=1)
    final_data.sort_values(by=['alignment_score'], ascending=False, inplace=True)
    final_data.drop_duplicates(subset=['Aptamers'], inplace=True)
    final_data['alignment_score'] = final_data['alignment_score'].apply(lambda x: str(round(x * 100, 2)) + "%")
    
    return final_data[['Aptamers', 'alignment_score']].reset_index(drop=True)