import argparse
import numpy as np
from itertools import groupby
import pandas as pd


SCORE = 0

def fastaread(fasta_name):
    """
    Read a fasta file. For each sequence in the file, yield the header and the actual sequence.
    In Ex1 you may assume the fasta files contain only one sequence.
    You may keep this function, edit it, or delete it and implement your own reader.
    """
    f = open(fasta_name)
    faiter = (x[1] for x in groupby(f, lambda line: line.startswith(">")))
    for header in faiter:
        header = next(header)[1:].strip()
        seq = "".join(s.strip() for s in next(faiter))
        return header, seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_a',
                        help='Path to first FASTA file (e.g. fastas/HomoSapiens-SHH.fasta)')
    parser.add_argument('seq_b', help='Path to second FASTA file')
    parser.add_argument('--align_type', help='Alignment type (e.g. local)',
                        required=True)
    parser.add_argument('--score',
                        help='Score matrix in.tsv format (default is score_matrix.tsv) ',
                        default='score_matrix.tsv')
    command_args = parser.parse_args()
    header_a, seq_a = fastaread(command_args.seq_a)
    header_b, seq_b = fastaread(command_args.seq_b)
    score_df = pd.read_csv(command_args.score, sep='\t', index_col=0)
    # fasta_head, fasta_seq = fastaread(command_args)
    if command_args.align_type == 'global':
        alignment, score = global_seq(seq_a, seq_b, score_df)
    elif command_args.align_type == 'local':
        alignment, score = local_seq(seq_a, seq_b, score_df)
    elif command_args.align_type == 'overlap':
        alignment, score = overlap_seq(seq_a, seq_b, score_df)
    print((alignment, score))


def sigma(score_df, seq_a, seq_b, i, j):
    if (i == '-'):
        a = '-'
    else:
        a = seq_a[i]
    if (j == '-'):
        b = '-'
    else:
        b = seq_b[j]
    return score_df[a][b]


def global_seq(seq_a, seq_b, score_df):
    n = len(seq_a)
    m = len(seq_b)
    mat = np.ndarray(shape=(n, m), dtype=tuple)
    mat[0][0] = (0, (0, 0))
    for col in range(m):
        for row in range(n):
   #         print((row,col))
            if row == 0 and col == 0:
                mat[row][col] = (0, (0, 0))
            elif row != 0 and col != 0:
                o1 = mat[row-1][col][SCORE] + sigma(score_df, seq_a, seq_b, row, '-')
                o2 = mat[row][col-1][SCORE] + sigma(score_df, seq_a, seq_b, '-', col)
                o3 = mat[row-1][col-1][SCORE] + sigma(score_df, seq_a, seq_b, row, col)
                print(max(o1, o2, o3))
                idx = np.argmax(np.array([o1, o2, o3]))
                if idx == 0:
                    mat[row][col] = (o1, (row-1, col))
                if idx == 1:
                    mat[row][col] = (o2, (row, col-1))
                if idx == 2:
                    mat[row][col] = (o3, (row-1, col-1))
            elif col == 0:
                o1 = mat[row-1][col][SCORE] + sigma(score_df, seq_a, seq_b, row, '-')
                mat[row][col] = (o1, (row-1, col))
            elif row == 0:
                o1 = mat[row][col-1][SCORE] + sigma(score_df, seq_a, seq_b, '-', col)
                mat[row][col] = (o1, (row, col-1))
    return get_alignment_result(seq_a, seq_b, mat, n, m)

def get_alignment_result(seq_a, seq_b, mat, n, m):
    score = mat[n-1][m-1][0]
    coord = (n-1, m-1)
    alignment_a = ''
    alignment_b = ''
    while coord != (0, 0):
        prev_coord = mat[coord[0]][coord[1]][1]
        # diagonal case
        if (prev_coord[0] == coord[0] - 1) and (prev_coord[1] == coord[1] - 1):
            alignment_a += seq_a[coord[0]]
            alignment_b += seq_b[coord[1]]
        # horizontal case - progressed in seq_b but not in seq_a
        elif (prev_coord[0] == coord[0] - 1) and (
                prev_coord[0] != coord[0] - 1):
            alignment_a += '_'
            alignment_b += seq_b[coord[1]]
        # vertical case - progressed in seq_a but not in seq_b
        elif (prev_coord[0] != coord[0] - 1) and (
                prev_coord[0] == coord[0] - 1):
            alignment_a += seq_a[coord[0]]
            alignment_b += '_'
        coord = prev_coord
    return (alignment_a[::-1], alignment_b[::-1]), score


def local_seq(seq_a, seq_b, score_df):
    pass


def overlap_seq(seq_a, seq_b, score_df):
    pass


if __name__ == '__main__':
    main()
