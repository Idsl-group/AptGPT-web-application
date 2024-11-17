import os
import RNA
import logging
from arnie.mfe import mfe


def make_image(sequence, structure, file):
    RNA.svg_rna_plot(sequence, structure, file)
    return file

def create_secondary_structure_images(df, sc=True):
    logging.info('Loading secondary structure image module')
    
    if sc:
        logging.info('Extracting secondary structure for each aptamer')
        df['Secondary Structure'] = df['Aptamers'].apply(lambda x: mfe(x))

    logging.info('Creating secondary structure images')
    df['Image Paths'] = df.apply(lambda row: make_image(row['Aptamers'], row['Secondary Structure'], file=f'{os.path.abspath(os.getcwd())}/images/SecondaryStructure_{row["Aptamers"]}.svg'), axis=1)

    return df