import numpy as np
import os
import pandas as pd
import torch
import esm
import collections


def esm_embeddings(peptide_sequence_list):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using {device} device")
    # NOTICE: ESM for embeddings is quite RAM usage, if your sequence is too long,
    #         or you have too many sequences for transformation in a single converting,
    #         you conputer might automatically kill the job.

    # load the model
    # NOTICE: if the model was not downloaded in your local environment, it will automatically download it.
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    # model = model.to(device)
    model.eval()  # disables dropout for deterministic results

    # load the peptide sequence list into the bach_converter
    batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list)
    # batch_tokens=batch_tokens.to(device)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    ## batch tokens are the embedding results of the whole data set

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        # Here we export the last layer of the EMS model output as the representation of the peptides
        # model'esm2_t6_8M_UR50D' only has 6 layers, and therefore repr_layers parameters is equal to 6
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
    # save dataset
    # sequence_representations is a list and each element is a tensor
    embeddings_results = collections.defaultdict(list)
    for i in range(len(sequence_representations)):
        # tensor can be transformed as numpy sequence_representations[0].numpy() or sequence_representations[0].to_list
        each_seq_rep = sequence_representations[i].tolist()
        for each_element in each_seq_rep:
            embeddings_results[i].append(each_element)
    embeddings_results = pd.DataFrame(embeddings_results).T
    return embeddings_results


# %%
def gen_esm_feature0(inputpath, resultpath):
    inputpath = inputpath
    sequence_list = get_seq(inputpath)

    embeddings_results = pd.DataFrame()
    i = 1
    for seq in sequence_list:
        print('The' + str(i) + 'sequence starts compiling')
        format_seq = [seq, seq]  # the setting is just following the input format setting in ESM model, [name,sequence]
        tuple_sequence = tuple(format_seq)
        peptide_sequence_list = []
        peptide_sequence_list.append(
            tuple_sequence)  # build a summarize list variable including all the sequence information
        # employ ESM model for converting and save the converted data in csv format
        one_seq_embeddings = esm_embeddings(peptide_sequence_list)
        embeddings_results = pd.concat([embeddings_results, one_seq_embeddings])
        print('The' + str(i) + 'sequence completes compilation')
        i += 1

    embeddings_results.to_csv(resultpath)


def esm_embeddings1(peptide_sequence_list, initial_trim_length=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)  # Move the model to GPU
    model.eval()  # Ensure that the model is in evaluation mode

    embeddings_results = []

    for name, seq in peptide_sequence_list:
        trimmed = False
        while True:
            try:
                # Attempt to transform the sequence and extract embeddings
                batch_labels, batch_strs, batch_tokens = batch_converter([(name, seq)])
                batch_tokens = batch_tokens.to(device)
                batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33])
                    token_representations = results["representations"][33]
                    seq_repr = token_representations[0, 1:batch_lens[0] - 1].mean(0)
                    embeddings_results.append((name, seq_repr.cpu().tolist()))
                    torch.cuda.empty_cache()
                break  # After success, jump out of the loop
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if len(seq) > initial_trim_length:
                        print(f"the sequence {name} is too long，attempting to truncate...")
                        seq = seq[:-initial_trim_length]  # Truncate a certain length from the end of the sequence
                        trimmed = True
                    else:
                        raise Exception("The sequence is too short to be further truncated") from e
                else:
                    raise e  # If the error is not a memory issue, throw it directly

        if trimmed:
            print(f"The sequence {name} has been successfully truncated and processed")

    return embeddings_results


def get_seq(path):
    from Bio.SeqIO import parse
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq

    file = open(path)
    records = parse(file, "fasta")  # Parse the content of the sequence file and return it as a list of SeqRecord objects
    # Use Python for loop traversal of records and print the properties of the SQL Record, such as ID, name, description, sequence data, etc
    seqs = []
    names = []
    cnt = 0
    greater_512 = 0
    for record in records:
        names.append(str(record.name) + str(cnt))
        tmp = str(record.seq)
        len(tmp)
        # print('tmp:',tmp)
        seqs.append(tmp)
        cnt = cnt + 1
    seq_dict = seqs
    print(greater_512)
    print(len(seq_dict))
    return seq_dict


def gen_esm_feature1(inputpath, resultpath):
    inputpath = inputpath
    sequence_list = get_seq(inputpath)

    embeddings_results = []
    i = 1
    for seq in sequence_list:
        print(f'The {i} sequence starts compiling')
        format_seq = [seq, seq]
        peptide_sequence_list = [format_seq]
        # Use ESM model for conversion
        one_seq_embeddings = esm_embeddings1(peptide_sequence_list)
        embeddings_results.extend(one_seq_embeddings)
        print(f'The {i} sequence starts compiling')

        i += 1

    # Convert the accumulated result list into a DataFrame
    embeddings_df = pd.DataFrame(embeddings_results, columns=['Name', 'Embedding'])
    embeddings_df.to_csv(resultpath)


# training dataset loading,
input = '../Data/Miyata/90/90_neg.fasta'
output = '../Features/Feature_csv/90_neg_esm1b.csv'

gen_esm_feature1(input, output)
