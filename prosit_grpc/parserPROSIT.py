#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parserPROSIT.py is a client converting Novor (or similar csv) output and converting it into an appropriate format
                for PROSIT predictions obtained by the predicPROSIT.py client.

__author__ = "Daniela Andrade Salazar"
__email__ = "daniela.andrade@tum.de"
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import pandas as pd
import os
import glob
import numpy as np
import itertools
import math

import prosit_grpc_client.__constants__ as C  # For the constants
from prosit_grpc_client.__utils__ import map_peptide_to_numbers, flatten_list, indices_to_one_hot, fill_zeros

#BATCH_SIZE = 6000 # max batch size for PROSIT

class ParserForPROSIT:

    def __init__(self, input_path: str, source,
                 batch_size: Optional[int] = 6000,
                 estimated_ce: Optional[float] = 0.3):
        """
        :param input_path: either path or folder or list of paths. 
                           If folder is given all csv-files from that directory will be considered for Novor
                           and txt-files for MaxQuant
        :param columns: comma-separated list containing columns to be found in the csv files, MANDATORY COLUMNS are peptide (sequence), z (charge),  and scanNum
                        charges larger than 6 will be set to 6
        """
        implemented = ('novor', 'maxquant')
        if source not in implemented:
            raise ValueError("Source '{}' not implemented yet. Available: ".format(source, implemented))
        self.source=source
        self.estimated_ce = estimated_ce
        self.batch_size = batch_size
        
        # Handle Input

        if isinstance(input_path, list):
            self.filepaths = [f for f in input_path if os.path.isfile(f)]
        elif os.path.isfile(input_path):
            self.filepaths = [input_path]
        elif os.path.isdir(input_path):
            ext = ".csv" if source=="novor" else ".txt"
            self.filepaths = glob.glob(input_path + "/*{}".format(ext))
        else:
            print(input_path)
            raise ValueError("Input path is not a directory of existent file")
        self.filenames = [os.path.basename(filepath) for filepath in self.filepaths]
        self.num_files = len(self.filepaths)
        print("[INFO] Number of input files to handle:", self.num_files)

        self.sequences_list = None
        self.charges_list = None
        self.collision_energies_list = None
        self.scan_nums_list = None
        self.info_batches = None
        self.labels = None
        
    class PSM:

        def __init__(self, filepath, source, batch_size, estimated_ce):
            """ PSM represents Peptide Spectrum Matching coming from either MaxQuant(DB) or Novor(De Novo)
            parameters are filepath, filename and columns of the csv file
            
            :param source: should be 'novor' or 'de novo' or 'db' or "maxquant"
            """            
            self.batch_size = batch_size
            self.estimated_ce = estimated_ce
            self.source = source
            self.filepath = filepath
            
            self.df = (self.set_df_novor() if source=="novor" else self.set_df_mq())[["id","scanNum", "z", "peptide", "Label"]]
            # Change scanNum to filename_scanNum
            filename = str(os.path.basename(filepath)).split(".")[0]
            self.df["scanNum"] = self.df["scanNum"].apply(lambda x: filename + "_scans_" + str(x) ) 
            
            self.num_seqs = len(self.df)
            self.num_batches = math.ceil(self.num_seqs / self.batch_size)
            print("""[INFO] File <{f}> has {s} PSMs and will be splitted into {b}
                        batches""".format(f=os.path.basename(filepath), s=self.num_seqs, b=self.num_batches))
            self.charges = None
            self.sequences = None
            self.collision_energies = None
            self.scan_nums = None
            self.ids = None

        def set_df_novor(self):
            novor_columns = ['id','scanNum','RT','mz(data)','z','pepMass(denovo)',
                             'err(data-denovo)','ppm(1e6*err/(mz*z))','score','peptide','aaScore']
            df = pd.read_csv(self.filepath, skip_blank_lines=True, comment="#", names=novor_columns)
            df["Label"] = 1
            
            ### filter invalid 
            #df =  df[(df["Modifications"]=="Unmodified") | (df["Modifications"]=="Oxidation (M)") | (df["Modifications"]=="2 Oxidation (M)")]
            #df = df[df["Length"]<=C.SEQ_LEN]
            df = df[~(df["peptide"].str.contains("U")) & ~(df["peptide"].str.contains("O")) ]
            df = df[df["z"]<= C.MAX_CHARGE]
            return df
        
        def set_df_mq(self):
            df = pd.read_csv(self.filepath, sep="\t", header=0)
            # Rename Columns to match novor columns
            cols_mapping = {"Scan number":"scanNum", "Charge":"z", "Sequence": "peptide", "id":"id_msms"}
            df.rename(columns=cols_mapping, inplace=True)
            old_len = len(df)
            ### Add Labels 
            df["Label"] = df["Reverse"].apply(lambda x: -1 if x =="+" else 1)
            
            ### filter invalid 
            df =  df[(df["Modifications"]=="Unmodified") | (df["Modifications"]=="Oxidation (M)") | (df["Modifications"]=="2 Oxidation (M)")]
            df = df[df["Length"]<=C.SEQ_LEN]
            df = df[~(df["peptide"].str.contains("U")) & ~(df["peptide"].str.contains("O")) ]
            df = df[df["z"]<= C.MAX_CHARGE]
            df = df[df["TYPE"] != 'MULTI-SECPEP']
            
            ### Keep only best targets and worst decoys
            targets = df[df["Label"]==1].sort_values(['scanNum', 'Score'], ascending=False)
            targets["duplicated"] = targets.duplicated('scanNum', keep='first')
            targets = targets[~targets["duplicated"]]

            decoys = df[df["Label"]!=1].sort_values(['scanNum', 'Score'], ascending=True)
            decoys["duplicated"] = decoys.duplicated('scanNum', keep='first')
            decoys = decoys[~decoys["duplicated"]]
            
            df = pd.concat([targets, decoys]).sort_index().drop(columns="duplicated")
            
            #df = df[df["Label"]==1].sort_values(['scanNum', 'Score'], ascending=False)
            #df["duplicated"] = df.duplicated('scanNum', keep='first')
            #df = df[~targets["duplicated"]].sort_index().drop(columns="duplicated")
            
            ### ROW ID AS INDEX
            df = df.reset_index(drop=False)
            df.rename(columns={"index":"id"}, inplace=True)
            #df.rename(columns={"Scan index" : "id"}, inplace=True)
            new_len = len(df)
            len_t = len(df[df["Label"]==1])
            len_d = new_len - len_t
            print("[INFO] Parsed Info from MQ. Deleted {} invalid PSMs and kept {} ({} T / {} D)".format(old_len - new_len, 
                                                                                                         new_len,
                                                                                                         len_t, len_d
                                                                                                        ))
            return df
         
        def peptides_to_numbers(self):
            """
            Converts numerical sequences to numerical with the help of the dictionary ALPHABET
            :param fixed_length: length of the the sequences, fill with zeros if sequence shorter than fixes length
            """
            # Convert to list of numbers
            self.df["peptide_num"] = self.df["peptide"].map(map_peptide_to_numbers)
            self.df["peptide_num"] = self.df["peptide_num"].map(lambda x: fill_zeros(x, C.SEQ_LEN))
                
            # Check if a sequence is too long
            old_num_seq = self.num_seqs
            self.df = self.df[self.df["peptide_num"].map(len)==C.SEQ_LEN]
            self.num_seq = len(self.df)  # Update length
            print("""\t [INFO] Lenght Check: Number of deleted sequences due to length greater than {}: 
                        {} out of {}""".format(C.SEQ_LEN, old_num_seq - self.num_seqs , old_num_seq))

        def charges_to_one_hot(self):
            """
            One hot encoding of every charge value --> 6 possible charges for every sequence for PROSIT
            """
            old_num_seq = self.num_seqs
            self.df = self.df[self.df.z <= C.MAX_CHARGE]
            self.num_seqs = len(self.df)  # Update length
            print("""\t [INFO] Charge Check: Number of deleted sequences due to charge larger than {}: 
                        {} out of {}""".format(C.MAX_CHARGE, old_num_seq - self.num_seqs , old_num_seq))
            
            self.df["z_one_hot"] = self.df["z"].map(lambda x: indices_to_one_hot(x, C.MAX_CHARGE))
        
        def split_batches(self):
            seqs = []
            charges = []
            ces = []
            scan_nums = []
            ids = []
            labels = []
            for i in range(self.num_batches):
                seqs.append(self.sequences[i*self.batch_size*C.SEQ_LEN: (i+1)*self.batch_size*C.SEQ_LEN])
                charges.append(self.charges[i*self.batch_size*6: (i+1)*self.batch_size*6])
                ces.append(self.collision_energies[i*self.batch_size: (i+1)*self.batch_size])
                scan_nums.append(self.scan_nums[i*self.batch_size: (i+1)*self.batch_size])
                ids.append(self.ids[i*self.batch_size: (i+1)*self.batch_size])
                labels.append(self.labels[i*self.batch_size: (i+1)*self.batch_size])
            return seqs, charges, ces, scan_nums, ids, labels
        
        def get_scan_nums(self):
            return list(self.df["scanNum"].apply(str))
        
        def get_ids(self):
            return list(self.df["id"])
        
        def get_labels(self):
            return list(self.df["Label"])
     
        def set_prosit_input(self):
            """
            Handle sequences, charges and collision energies for PROSIT input:
            Convert sequences to numbers, arrange to match fixed length and concatenate lists
            Convert charges to one-hot-encoding, delete rows with invalid charge, concatenate lists
            Collision energies currently hard coded to 0.3
            """
            self.peptides_to_numbers()  # set num sequences in df
            self.charges_to_one_hot()  # set charges in df
            # convert to lists
            self.sequences = flatten_list(list(self.df["peptide_num"]))
            self.charges = flatten_list(list(self.df["z_one_hot"]))
            self.collision_energies = [self.estimated_ce]*self.num_seqs  # currently hardcoded
            self.scan_nums = self.get_scan_nums()
            self.labels = self.get_labels()
            self.ids = self.get_ids()
            
    def prepare_prosit_input(self):
        """
        Handle all PROSIT inputs
        Concatenate results for separate csv files into one 
        --> list of lists as results for sequences, collision energies and charges
        """
        self.sequences_list = []
        self.charges_list = []
        self.collision_energies_list = []
        self.scan_nums_list = []
        self.ids_list = []
        self.labels_list = []
        
        self.info_batches = {}

        for i in range(len(self.filepaths)):
            print("[INFO] Handling {}/{} input file(s)".format(i+1, self.num_files))
           
            psm = self.PSM(filepath=self.filepaths[i], source=self.source, 
                     batch_size=self.batch_size, estimated_ce=self.estimated_ce)
            print("\t [INFO] Done reading input file {}".format(self.filenames[i]))
            
            psm.set_prosit_input()
            print("\t [INFO] Done setting PROSIT input for current file having {} sequences".format(psm.num_seqs))
            
            batches = psm.split_batches()
            self.sequences_list += batches[0]
            self.charges_list += batches[1]
            self.collision_energies_list += batches[2]
            self.scan_nums_list += batches[3] # append scanNums
            self.ids_list += batches[4]
            self.labels_list += batches[5]
            
            # Add info on batches:
            self.info_batches[self.filenames[i]] = psm.num_batches
        print("[INFO] Done with all files.")
        
    def write_prosit_to_csv(self, dfs, path_to_folder=""):
        """
        This function gets the result of predictPROSIT as dictionary idx_batch : df
        where df is the resulted dataframe from PROSIT 
        --> as output it returns the same number of input files csv with the spectra only containing the PROSIT predictions
        """
        # Create Ouput dir if not existent 
        path_to_folder = os.path.join(".",path_to_folder)
        if not os.path.exists(path_to_folder):
            os.mkdir(path_to_folder)
        idx_dfs = 0
        # index for dfs to concatenate
        for i in range(len(self.filepaths)):
            filepath = self.filepaths[i]
            filename = self.filenames[i]
            num_batches = self.info_batches[filename]
            
            df = None
            start = idx_dfs
            stop = idx_dfs+num_batches
            print("[INFO] File {f} has {b} batches and takes dfs from {frm} to {to}".format(f=filename, 
                                                                                            b=num_batches, 
                                                                                            frm=start, 
                                                                                            to=stop-1))
            for j in range(start, stop):
                df = dfs[j] if df is None else pd.concat([df, dfs[j]], ignore_index=True)
            
            filepath = os.path.join(path_to_folder,
                                    "{name}.prosit.csv".format(name=filename.split(".")[0]))
            df.to_csv(filepath, index=None)
            print("[INFO] Wrote predictions for dataset {idx} to <{path}>".format(idx=i, path=filepath))
            idx_dfs = stop
            ## Concatenate dfs from index i to i+num_batches
