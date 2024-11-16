import io
from Bio import SeqIO
from arnie.mfe import mfe
from .optimal_aptamer_finder.Sequence import *

def read_fastq(fastq_file):
    data = []
    fastq_file.seek(0)
    
    # Read the FASTQ file using Biopython
    file = io.TextIOWrapper(fastq_file, encoding='utf-8')
    for record in SeqIO.parse(file, "fastq"):
        seq_id = record.id
        sequence = str(record.seq)[18:48]

        try:
            structure = mfe(sequence)
        except Exception as e:
            print(f"Error computing MFE for sequence {seq_id}: {e}")
            structure = None

        # Append the results to the list
        data.append({
            'sequence_id': seq_id,
            'sequence': sequence,
            'mfe_structure': structure
        })

    df = pd.DataFrame(data, columns=['sequence_id', 'sequence', 'mfe_structure'])

    return df


def optimal_aptamer_finder_clustering(binding_target, rounds, data, threshold, forward_primer, reverse_primer):
    sequencelib = SequenceLibrary(with_primers=False, primers=(forward_primer, reverse_primer))
    sequencelib.add_data(str(binding_target), str(rounds), data.sequence.values, structure_mode='from_ct', top_k=None)
    with open(f'./results/{TIME}/sequencelib.pickle', 'wb') as f:
        pickle.dump(sequencelib, f)
        f.close()
    get_priority_clusters(alignment_threshold=threshold, primers=(forward_primer, reverse_primer))
    clustered_data = generate_all_recommendations([binding_target], [str(rounds)], 1.0)
    clustered_data = pd.DataFrame(clustered_data, columns=['Index', 'Sequences', 'Secondary structure', 'Scores', 'Split'])

    return clustered_data.head(20)