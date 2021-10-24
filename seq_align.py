import argparse
import numpy as np
from itertools import groupby
import pandas as pd

SCORE = 0
COORD = 1
GAP = '-'
X = 0
Y = 1


class GlobalAlignment:
    def __init__(self, score_df, seq_a, seq_b):
        self.score_df = score_df
        self.seq_a = seq_a
        self.seq_b = seq_b

        self.n = len(seq_a)
        self.m = len(seq_b)
        self.mat = np.ndarray(shape=(self.n+1, self.m+1), dtype=tuple)

    def sigma(self, i, j):
        # if i is not gap, gets corresponding letter from seq
        if i == GAP:
            a = i
        else:
            a = self.seq_a[i-1]
        # if j is not gap, gets corresponding letter from seq
        if j == GAP:
            b = j
        else:
            b = self.seq_b[j-1]
        return self.score_df[a][b]

    def align(self):
        for col in range(self.m+1):
            for row in range(self.n+1):
                # base case, assign score = 0 and prev coordinates 0,0
                if row == 0 and col == 0:
                    self.mat[row][col] = (0, (0, 0))
                # first column has only the option of gaps
                elif col == 0:
                    opt_score = self.mat[row-1][col][SCORE] + self.sigma(row, GAP)
                    self.mat[row][col] = (opt_score, (row-1, col))
                # first row has only the option of gaps
                elif row == 0:
                    opt_score = self.mat[row][col - 1][SCORE] + self.sigma(GAP, col)
                    self.mat[row][col] = (opt_score, (row, col - 1))
                # all other blocks have three options
                else:
                    # for each of the three options create a tuple of score, coord
                    options = [
                    (self.mat[row-1][col][SCORE] + self.sigma(row, GAP), (row-1, col)),
                    (self.mat[row][col-1][SCORE] + self.sigma(GAP, col), (row, col-1)),
                    (self.mat[row-1][col-1][SCORE] + self.sigma(row, col), (row-1, col-1))]

                    # get optimal tuple based on max score
                    optimal = max(options, key=lambda option: option[SCORE])
                    self.mat[row][col] = optimal

    def get_strings(self):
        coord = (self.n, self.m)
        alignment_a = ''
        alignment_b = ''
        while coord != (0, 0):
            prev_coord = self.mat[coord[X]][coord[Y]][COORD]
            # diagonal case
            if (prev_coord[X] == coord[X] - 1) and \
                    (prev_coord[Y] == coord[Y] - 1):
                alignment_a += self.seq_a[coord[X]-1]
                alignment_b += self.seq_b[coord[Y]-1]
            # horizontal case - progressed in seq_b but not in seq_a
            elif (prev_coord[X] == coord[X] - 1) and \
                    (prev_coord[Y] != coord[Y] - 1):
                alignment_a += self.seq_a[coord[X]-1]
                alignment_b += GAP
            # vertical case - progressed in seq_a but not in seq_b
            elif (prev_coord[X] != coord[X] - 1) and \
                    (prev_coord[Y] == coord[Y] - 1):
                alignment_a += GAP
                alignment_b += self.seq_b[coord[Y]-1]
            coord = prev_coord
        # alignment is recovered in reverse order, [::-1] reverses strings back
        return alignment_a[::-1], alignment_b[::-1]

    def get_score(self):
        return self.mat[self.n][self.m][SCORE]



class LocalAlignment:
    def __init__(self, score_df, seq_a, seq_b):
        self.score_df = score_df
        self.seq_a = seq_a
        self.seq_b = seq_b

        self.n = len(seq_a)
        self.m = len(seq_b)
        self.mat = np.ndarray(shape=(self.n + 1, self.m + 1), dtype=tuple)

    def sigma(self, i, j):
        # if i is not gap, gets corresponding letter from seq
        if i == GAP:
            a = i
        else:
            a = self.seq_a[i - 1]
        # if j is not gap, gets corresponding letter from seq
        if j == GAP:
            b = j
        else:
            b = self.seq_b[j - 1]
        return self.score_df[a][b]

    def align(self):
        for col in range(self.m + 1):
            for row in range(self.n + 1):
                # base case, assign score = 0 and no prev coordinates
                if row == 0 or col == 0:
                    self.mat[row][col] = (0, None)
                else:
                    # for each of the three options create a tuple of score, coord
                    options = [
                        (self.mat[row - 1][col][SCORE] + self.sigma(row, GAP),
                         (row - 1, col)),
                        (self.mat[row][col - 1][SCORE] + self.sigma(GAP, col),
                         (row, col - 1)),
                        (self.mat[row - 1][col - 1][SCORE] + self.sigma(row,
                                                                        col),
                         (row - 1, col - 1)),
                        (0, None)]

                    # get optimal tuple based on max score
                    optimal = max(options, key=lambda option: option[SCORE])
                    self.mat[row][col] = optimal

    def get_strings(self):
        # get coordinates of maximum score
        mat_df = pd.DataFrame(self.mat)
        score_mat = mat_df.applymap(lambda x: x[SCORE]).to_numpy()
        coord = np.unravel_index(score_mat.argmax(), score_mat.shape)
        prev_coord = self.mat[coord[X]][coord[Y]][COORD]
        alignment_a = ''
        alignment_b = ''
        while prev_coord != None:
            # diagonal case
            if (prev_coord[X] == coord[X] - 1) and \
                    (prev_coord[Y] == coord[Y] - 1):
                alignment_a += self.seq_a[coord[X] - 1]
                alignment_b += self.seq_b[coord[Y] - 1]
            # horizontal case - progressed in seq_b but not in seq_a
            elif (prev_coord[X] == coord[X] - 1) and \
                    (prev_coord[Y] != coord[Y] - 1):
                alignment_a += self.seq_a[coord[X] - 1]
                alignment_b += GAP
            # vertical case - progressed in seq_a but not in seq_b
            elif (prev_coord[X] != coord[X] - 1) and \
                    (prev_coord[Y] == coord[Y] - 1):
                alignment_a += GAP
                alignment_b += self.seq_b[coord[Y] - 1]
            coord = prev_coord
            prev_coord = self.mat[coord[X]][coord[Y]][COORD]
        # alignment is recovered in reverse order, [::-1] reverses strings back
        return alignment_a[::-1], alignment_b[::-1]


    def get_score(self):
        mat_df = pd.DataFrame(self.mat)
        score_df = mat_df.applymap(lambda x: x[SCORE])
        return score_df.to_numpy().max()


class OverlapAlignment:
    def __init__(self, score_df, seq_a, seq_b):
        self.score_df = score_df
        self.seq_a = seq_a
        self.seq_b = seq_b

        self.n = len(seq_a)
        self.m = len(seq_b)
        self.mat = np.ndarray(shape=(self.n+1, self.m+1), dtype=tuple)


    def sigma(self, i, j):
        # if i is not gap, gets corresponding letter from seq
        if i == GAP:
            a = i
        else:
            a = self.seq_a[i-1]
        # if j is not gap, gets corresponding letter from seq
        if j == GAP:
            b = j
        else:
            b = self.seq_b[j-1]
        return self.score_df[a][b]


    def align(self):
        for col in range(self.m+1):
            for row in range(self.n+1):
                # base case, assign score = 0 and prev coordinates 0,0
                if row == 0 and col == 0:
                    self.mat[row][col] = (0, (0, 0))
                # first column has only the option of gaps
                elif col == 0:
                    opt_score = self.mat[row-1][col][SCORE] + max(0,self.sigma(row, GAP))
                    self.mat[row][col] = (opt_score, (row-1, col))
                # first row has only the option of gaps
                elif row == 0:
                    opt_score = self.mat[row][col - 1][SCORE] + self.sigma(GAP, col)
                    self.mat[row][col] = (opt_score, (row, col - 1))
                # all other blocks have three options
                elif row == self.n:
                    options = [
                        (self.mat[row-1][col][SCORE] + max(0, self.sigma(row, GAP)), (row-1, col)),
                        (self.mat[row][col-1][SCORE] + max(0, self.sigma(GAP, col)), (row, col-1)),
                        (self.mat[row-1][col-1][SCORE] + max(0,self.sigma(row, col)), (row-1, col-1))]
                    # get optimal tuple based on max score
                    optimal = max(options, key=lambda option: option[SCORE])
                    self.mat[row][col] = optimal
                else:
                    # for each of the three options create a tuple of score, coord
                    options = [
                        (self.mat[row-1][col][SCORE] + self.sigma(row, GAP), (row-1, col)),
                        (self.mat[row][col-1][SCORE] + self.sigma(GAP, col), (row, col-1)),
                        (self.mat[row-1][col-1][SCORE] + self.sigma(row, col), (row-1, col-1))]

                    # get optimal tuple based on max score
                    optimal = max(options, key=lambda option: option[SCORE])
                    self.mat[row][col] = optimal


    def get_strings(self, coord):
        coord = coord
        alignment_a = ''
        alignment_b = ''
        while coord != (0, 0):
            prev_coord = self.mat[coord[X]][coord[Y]][COORD]
            # diagonal case
            if (prev_coord[X] == coord[X] - 1) and \
                    (prev_coord[Y] == coord[Y] - 1):
                alignment_a += self.seq_a[coord[X]-1]
                alignment_b += self.seq_b[coord[Y]-1]
            # horizontal case - progressed in seq_b but not in seq_a
            elif (prev_coord[X] == coord[X] - 1) and \
                    (prev_coord[Y] != coord[Y] - 1):
                alignment_a += self.seq_a[coord[X]-1]
                alignment_b += GAP
            # vertical case - progressed in seq_a but not in seq_b
            elif (prev_coord[X] != coord[X] - 1) and \
                    (prev_coord[Y] == coord[Y] - 1):
                alignment_a += GAP
                alignment_b += self.seq_b[coord[Y]-1]
            coord = prev_coord
        # alignment is recovered in reverse order, [::-1] reverses strings back
        return alignment_a[::-1], alignment_b[::-1]



    def get_best(self):
        mat_df = pd.DataFrame(self.mat[self.n][:])
        score_df = mat_df.applymap(lambda x: x[SCORE])
        scores = score_df.to_numpy()
        coords = (self.n, scores.argmax())
        return coords


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


def print_alignment(str_a, str_b, type, score):
    # prints alignment strings in chunks of 50 chars
    for i in range(0, max(len(str_a), len(str_b)), 50):
        print(str_a[i:i + 50])
        print(str_b[i:i + 50])
        print()
    print(f'{type}:{score}')

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
        g = GlobalAlignment(score_df, seq_a, seq_b)
        g.align()
        score = g.get_score()
        str_a, str_b = g.get_strings()
        print_alignment(str_a, str_b, command_args.align_type, score)
    elif command_args.align_type == 'local':
        l = LocalAlignment(score_df, seq_a, seq_b)
        l.align()
        score = l.get_score()
        str_a, str_b = l.get_strings()
        print_alignment(str_a, str_b, command_args.align_type, score)
    elif command_args.align_type == 'overlap':
        o = OverlapAlignment(score_df, seq_a, seq_b)
        o.align()
        best = o.get_best()
        score = o.mat[best][SCORE]
        str_a, str_b = o.get_strings(best)
        print_alignment(str_a, str_b, command_args.align_type, score)



if __name__ == '__main__':
    main()
