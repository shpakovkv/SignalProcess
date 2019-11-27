#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

import os
import sys
import shutil


def main():
    assert len(sys.argv) > 2, "Specify input folder, osc output folder, screenshot output folder, setup output folder"
    source_dir = os.path.abspath(sys.argv[1])
    print("Input folder", source_dir)
    osc_dir = os.path.abspath(sys.argv[2])
    print("OSC output folder", osc_dir)
    if not os.path.isdir(osc_dir):
        os.makedirs(osc_dir)

    if len(sys.argv) > 3:
        screen_dir = os.path.abspath(sys.argv[3])
        if not os.path.isdir(screen_dir):
            os.makedirs(screen_dir)

    if len(sys.argv) > 4:
        setup_dir = os.path.abspath(sys.argv[4])
        if not os.path.isdir(setup_dir):
            os.makedirs(setup_dir)

    for x in os.walk(sys.argv[1]):
        for filename in x[2]:
            # for WFM
            if filename.upper().endswith('WFM'):
                new_name = os.path.basename(x[0]) + '_' + filename
                new_fullpath = os.path.join(osc_dir, new_name)
                print('Copy osc "{}/{}" to {}'.format(os.path.basename(x[0]), filename, new_fullpath))
                shutil.copy2(os.path.join(x[0], filename), new_fullpath)

            # for .png
            elif len(sys.argv) > 3 and filename.upper().endswith('PNG'):
                new_fullpath = os.path.join(screen_dir, filename)
                print('Copy screen "{}/{}" to {}'.format(os.path.basename(x[0]), filename, new_fullpath))
                shutil.copy2(os.path.join(x[0], filename), new_fullpath)

            # for .set
            elif len(sys.argv) > 4 and filename.upper().endswith('SET'):
                new_fullpath = os.path.join(setup_dir, filename)
                print('Copy setup "{}/{}" to {}'.format(os.path.basename(x[0]), filename, new_fullpath))
                shutil.copy2(os.path.join(x[0], filename), new_fullpath)
        # print('{} : {}'.format(os.path.basename(x[0]), x[2]))


if __name__ == '__main__':
    main()
