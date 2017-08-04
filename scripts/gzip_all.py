#!/usr/bin/env python

# Copyright (c) 2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

use_mpi = True
if use_mpi:
    from mpi4py import MPI
import os
import sys
import re
import traceback
import subprocess

def list_files(path, pattern):
    """
    Generate a list of files with full paths
    """

    files = []

    for root, directories, filenames in os.walk(path, followlinks=True):

        for filename in filenames:
            if pattern.match(filename):
                files.append(os.path.join(root, filename))

    return files

def main():

    if use_mpi:
        comm = MPI.COMM_WORLD
        ntask = comm.size
        rank = comm.rank
    else:
        ntask = 1
        rank = 0

    if len(sys.argv) != 3:
        if rank == 0:
            print('Usage: gzip_all <path> <regex pattern>')
        return

    path = sys.argv[1]
    pattern = sys.argv[2]

    regexp = re.compile(pattern)

    if rank == 0:
        if use_mpi: t1 = MPI.Wtime()
        print('Getting a list of files in {} matching {}'.format(path, pattern))
        files = sorted(list_files(path, regexp))
        if use_mpi:
            t2 = MPI.Wtime()
            print('List assembled in {:.2f} s. There are {} files to compress'
                  ''.format(t2-t1, len(files)), flush=True)
    else:
        files = None

    if use_mpi:
        files = comm.bcast(files)

    for i, fn in enumerate(files):
        if i % ntask != rank: continue
        if use_mpi: t1 = MPI.Wtime()
        print(rank, ': Compressing', fn, flush=True)
        subprocess.call(['gzip', fn])
        if use_mpi:
            t2 = MPI.Wtime()
            print(rank, ': Compressed {} in {:.2f} s ({} / {})'
                  ''.format(fn, t2-t1, i/ntask+1, len(files)/ntask+1),
                  flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Exception occurred: "{}"'.format(e), flush=True)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print('*** print_tb:')
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print('*** print_exception:')
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)
        print('*** print_exc:')
        traceback.print_exc()
        print('*** format_exc, first and last line:')
        formatted_lines = traceback.format_exc().splitlines()
        print(formatted_lines[0])
        print(formatted_lines[-1])
        print('*** format_exception:')
        print(repr(traceback.format_exception(exc_type, exc_value,
                                              exc_traceback)))
        print('*** extract_tb:')
        print(repr(traceback.extract_tb(exc_traceback)))
        print('*** format_tb:')
        print(repr(traceback.format_tb(exc_traceback)))
        print('*** tb_lineno:', exc_traceback.tb_lineno, flush=True)
        MPI.COMM_WORLD.Abort()
