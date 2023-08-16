#!/usr/bin/env python3
"""
Files duplicates analysis functions.

Author of algorithm: Todor Minakov from stackoverflow.com.

Modified by Shpakov Konstantin.
"""
from collections import defaultdict
import hashlib
import os
import sys

from typing import Iterable


CHUNK_SIZE = 1024
# HASH_FUNCTION = hashlib.sha1
HASH_FUNCTION = hashlib.sha256


def chunk_reader(fobj, chunk_size=CHUNK_SIZE):
    """Iterator that reads a file in chunks of bytes"""
    while True:
        chunk = fobj.read(chunk_size)
        if not chunk:
            return
        yield chunk


def get_hash(filename, first_chunk_only=False, hash=HASH_FUNCTION):
    """Returns the hash value of the first chunk or the entire file.

    :param filename: path to file
    :type filename: str
    :param first_chunk_only: switches the algorithm between analyzing
                             the entire file or just a first chunk
    :type first_chunk_only: bool
    :param hash: hash function
    :type hash: function
    :return: hash digest
    :rtype: bytes
    """
    hashobj = hash()
    file_object = open(filename, 'rb')

    if first_chunk_only:
        hashobj.update(file_object.read(CHUNK_SIZE))
    else:
        for chunk in chunk_reader(file_object):
            hashobj.update(chunk)
    hashed = hashobj.digest()

    file_object.close()
    return hashed


def get_file_dict_by_size(folder_list):
    """Groups files by file size.

    :param folder_list: a list of folder with files
    :type folder_list: list
    :return: a dict with file size as keys and a list of file paths as values
    :rtype: defaultdict
    """
    files_by_size = defaultdict(list)  # dict of size_in_bytes: [full_path_to_file1, full_path_to_file2, ]
    for path in folder_list:
        for dirpath, dirnames, filenames in os.walk(path):
            # get all files that have the same size - they are the collision candidates
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                try:
                    # if the target is a symlink (soft one), this will
                    # dereference it - change the value to the actual target file
                    full_path = os.path.realpath(full_path)
                    file_size = os.path.getsize(full_path)
                    files_by_size[file_size].append(full_path)
                except (OSError,):
                    # not accessible (permissions, etc) - pass on
                    continue

    return files_by_size


def group_by_first_chunk_hash_and_size(hashes_by_size):
    """Check files, that were grouped by file size
     and regroups that files by the first chunk hash and size.
     Skips files with unique size.

    :param hashes_by_size: a dict with file size as keys and a list of file paths as values
    :type hashes_by_size: dict or defaultdict
    :return: a dict with tuples of hashes of the first chunk and file size as keys
             and a list with file paths as values
    :rtype: defaultdict
    """

    # dict of (hash1k, size_in_bytes): [full_path_to_file1, full_path_to_file2, ]
    files_by_first_chunk_hash_and_size = defaultdict(list)
    for size_in_bytes, files in hashes_by_size.items():
        if len(files) < 2:
            continue    # this file size is unique, no need to spend CPU cycles on it

        for filename in files:
            try:
                small_hash = get_hash(filename, first_chunk_only=True)
                # the key is the hash on the first 1024 bytes plus the size - to
                # avoid collisions on equal hashes in the first part of the file
                # credits to @Futal for the optimization
                files_by_first_chunk_hash_and_size[(small_hash, size_in_bytes)].append(filename)
            except (OSError,):
                # the file access might've changed till the exec point got here
                continue
    return files_by_first_chunk_hash_and_size


def group_by_hash(hashes_on_1k, verbose=False):
    """Check files, that were grouped by the first chunk hash and size
     and regroups that files by full hash.
     Skips files with unique hash.

    :param hashes_on_1k: dict with a tuple of hash on 1 chunk and file size as keys
                         and a list with files paths as values
    :type hashes_on_1k: dict or defaultdict
    :param verbose: prints duplicates in pairs
    :type verbose: bool
    :return: a dict with hash as keys and a list of paths as values
    :rtype: defaultdict
    """
    files_by_hash = defaultdict(list)  # dict of full_file_hash: full_path_to_file_string
    for __, files_list in hashes_on_1k.items():
        if len(files_list) < 2:
            continue    # this hash of fist 1k file bytes is unique, no need to spend cpy cycles on it

        for filename in files_list:
            try:
                full_hash = get_hash(filename, first_chunk_only=False)
                duplicate = files_by_hash.get(full_hash)
                if duplicate:
                    files_by_hash[full_hash].append(filename)
                    if verbose:
                        print("Duplicate found: {} and {}".format(filename, duplicate))
                else:
                    files_by_hash[full_hash].append(filename)
            except (OSError,):
                # the file access might've changed till the exec point got here
                continue
    return files_by_hash


def print_duplicates(files_by_hash_dict):
    """Prints the full paths of duplicates.

    :param files_by_hash_dict: dict with hash as keys and a list of paths as values
    :type files_by_hash_dict: dict or defaultdict
    :return: None
    :rtype: None
    """
    dupl_groups = len(files_by_hash_dict.keys())
    if dupl_groups > 0:
        print(f"{dupl_groups} group(s) of DUPLICATES FOUND !!")
        idx = 1
        for _, item in files_by_hash_dict.items():
            print("--------------")
            print(f"Duplicates group {idx}:")
            print("\n".join(path for path in item))
            idx += 1
    else:
        print("No duplicates found.")


def check_for_duplicates(folder_list):
    """Check files from one or more folder for duplicates.
    Prints the full paths of duplicates.

    :param folder_list: list of folders to check files
    :type folder_list: Iterable
    :return: None
    :rtype: None
    """
    print("Checking files for duplicates...", end="")

    for path in folder_list:
        assert os.path.isdir(path), "Error! Can't find folder '{path}'"

    files_by_size = get_file_dict_by_size(folder_list)

    # For all files with the same file size, get their hash on the 1st 1024 bytes only
    files_by_hash1k_and_size = group_by_first_chunk_hash_and_size(files_by_size)

    # For all files with the hash on the 1st 1024 bytes, get their hash on the full file - collisions will be duplicates
    files_by_fullhash = group_by_hash(files_by_hash1k_and_size, verbose=False)

    print(" Done.")
    print_duplicates(files_by_fullhash)


if __name__ == "__main__":
    if sys.argv[1:]:
        check_for_duplicates(sys.argv[1:])
    else:
        print("Please pass the paths to check as parameters to the script")
        print("python duplicate_checker.py path/to/folder1 path/to/folder2 etc.")
